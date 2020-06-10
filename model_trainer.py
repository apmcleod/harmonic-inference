import os
import sys
import shutil

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from harmonic_inference_data import pad_and_collate_samples




class ModelTrainer():
    """
    A ModelTrainer object can be used to train or evaluate a pytorch model.
    """
    
    def __init__(self, model, train_dataset=None, valid_dataset=None, test_dataset=None,
                 device=torch.device("cpu"), seed=None,
                 batch_size=64, valid_batch_size=None, num_epochs=100, early_stopping=20,
                 optimizer=None, scheduler=None, schedule_var=None, criterion=None,
                 log_every=1, log_file=None,
                 save_every=10, save_dir='.', save_prefix='checkpoint',
                 resume=None):
        """
        Create a new ModelTrainer object.
        
        Parameters
        ----------
        model : torch.nn.Module
            The neural net to train or evaluate.
            
        train_dataset : torch.Dataset
            The training Dataset to load batches from. Required for train(), but not evaluate().
            
        valid_dataset : torch.Dataset
            The validation Dataset to load batches from. Required for train(), but not evaluate().
            
        test_dataset : torch.Dataset
            The test Dataset to load batches from. Required for evaluate(), but not train().
            
        device : torch.device
            The device to perform training or evaluation on.
            
        seed : int
            The random seed to seed torch.random with.
            
        batch_size : int
            The batch size to use for training.
            
        valid_batch_size : int
            The batch size to use for validation and evaluation. Defaults to batch_size if None.
            
        num_epochs : int
            The number of epochs to train for.
            
        early_stopping : int
            Stop before num_epochs if there has been no improvement in validation loss for this many epochs.
            
        optimizer : torch.optim
            The optimizer to use during training.
            
        scheduler : torch.scheduler
            The scheduler to use during training (if any).
            
        schedule_var : string
            The name of the variable to be passed to the scheduler (if any). Can be:
            valid_loss, valid_acc, train_loss, or train_acc.
            
        criterion : torch.criterion
            The loss criterion to use during training and evaluation.
            
        log_every : int
            Print to the log every this many epochs during training.
            
        log_file : file
            The file to print the log to. Prints to standard out if None.
            
        save_every : int
            Save a model checkpoint every this many epochs during training.
            
        save_dir : string
            The directory to save the model checkpoints to.
            
        save_prefix : string
            A prefix to use when saving the model checkpoint, by default.
            
        resume : string
            The path of a checkpoint file to load from, optionally.
        """
        if valid_batch_size is None:
            valid_batch_size = batch_size
            
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       collate_fn=pad_and_collate_samples) if train_dataset is not None else None
        self.valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False,
                                       collate_fn=pad_and_collate_samples) if valid_dataset is not None else None
        self.test_loader = DataLoader(test_dataset, batch_size=valid_batch_size, shuffle=False,
                                      collate_fn=pad_and_collate_samples) if test_dataset is not None else None
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.device = device
        self.model = model
        self.model.to(self.device)
        
        self.num_epochs = num_epochs
        self.epoch = 0
        self.early_stopping = early_stopping
        
        self.best_valid_acc = 0
        self.best_epoch = -1
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_var = schedule_var
        self.criterion = criterion
        
        if type(self.scheduler) is ReduceLROnPlateau:
            valid_vars = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc']
            assert self.schedule_var in valid_vars, (
                "If scheduler is ReduceLROnPlateau, schedule_var must be one of " +
                str(valid_vars)
            )
        
        self.log_every = log_every
        self.log_file = sys.stdout if log_file is None else log_file
        
        self.save_every = save_every
        self.save_dir = save_dir if save_dir is not None and save_dir != '' else '.'
        self.save_prefix = save_prefix if save_prefix is not None else 'checkpoint'
        os.makedirs(self.save_dir, exist_ok=True)
        
        if resume is not None:
            assert os.path.isfile(resume), f"Resume file {resume} not found (or not a file)."
            
            checkpoint = torch.load(resume)
            
            self.epoch = checkpoint['epoch']
            self.num_epochs += self.epoch
            
            self.best_valid_acc = checkpoint['best_valid_acc']
            self.best_epoch = checkpoint['best_epoch']
            
            if self.scheduler is not None:
                if 'scheduler' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                else:
                    print("Resume file has no scheduler information. Not loading it.", file=sys.stderr)
            elif 'scheduler' in checkpoint:
                print("Resume file has scheduler information saved, but scheduler is None. Not loading it.",
                      file=sys.stderr)
                
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)
            
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        
        
    def evaluate(self):
        """
        Get the loss and accuracy of the loaded model on the loaded test data.
        """
        # Temp copy of teset data into valid slot
        old_valid_loader = self.valid_loader
        self.valid_loader = self.test_loader
        loss, acc = iteration(train=False)
        self.valid_loader = old_valid_loader
        
        return loss, acc
        
        
    def train(self):
        """
        Train the loaded model on the loaded data.
        """
        for self.epoch in range(self.epoch, self.num_epochs):
            # Training
            train_loss, train_acc = self.iteration(train=True)

            # Validation
            valid_loss, valid_acc = self.iteration(train=False)
            
            if self.scheduler is not None:
                if type(self.scheduler) is ReduceLROnPlateau:
                    self.scheduler.step({
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'valid_loss': valid_loss,
                        'valid_acc': valid_acc
                    }[self.schedule_var])
                else:
                    self.scheduler.step()

            # Log printing
            if self.epoch % self.log_every == 0:
                self.log([self.epoch, train_loss, train_acc, valid_loss, valid_acc])

            # Save checkpoints
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
                self.best_epoch = self.epoch
            if self.epoch % self.save_every == 0:
                self.save(is_best=is_best, valid_acc=valid_acc)
                
            # Early stopping
            if self.early_stopping is not None:
                if self.best_epoch + self.early_stopping <= self.epoch:
                    print("Early stopping")
                    break
                
        # Always save last epoch
        self.save(is_best=is_best, valid_acc=valid_acc)
    
    
    
    def iteration(self, train=True):
        """
        Perform a single pass through a loaded Dataset and return the model's loss and accuracy.
        
        Parameters
        ----------
        train : bool
            If True, use self.train_loader and perform backprop. If False, use self.valid_loader and perform no backprop.
        """
        total_loss = 0
        total_acc = 0
        total_size = 0
        
        data_loader = self.train_loader if train else self.valid_loader
        
        torch.set_grad_enabled(train)
        
        for batch in data_loader:
            # Load data
            notes = batch['notes'].float()
            notes_lengths = batch['num_notes']
            labels = batch['chord']['one_hot'].long()
            this_batch_size = notes.shape[0]
            
            # Transfer to device
            notes, labels = notes.to(self.device), labels.to(self.device)
            
            if train:
                self.optimizer.zero_grad()
                
            outputs = self.model.forward(notes, notes_lengths)
            loss = self.criterion(outputs, labels)
            _, predictions = outputs.max(1)
            correct = (predictions == labels).sum().float()
            acc = correct / this_batch_size
            
            total_loss += this_batch_size * loss.item()
            total_acc += this_batch_size * acc.item()
            total_size += this_batch_size
            
            if train:
                loss.backward()
                self.optimizer.step()
                
        loss = total_loss / total_size
        acc = total_acc / total_size
        
        return loss, acc
    
    
    def log(self, log_vals):
        """
        Print the given values out to self.log_file.
        
        Parameters
        ----------
        log_val : list
            A list of values to print to the log file as comma-separated values.
        """
        print(','.join([str(val) for val in log_vals]), file=self.log_file)
    
    
    
    def save(self, is_best=False, valid_acc=None):
        """
        Save the currently loaded model to a file. By default,
        {self.save_dir}/{self.save_prefix}.pth.tar.
        
        Parameters
        ----------
        is_best : bool
            If True, this copies the resulting checkpoint file to best.pth.tar in the same directory.
            
        valid_acc : float
            The validation accuracy. Saved in the checkpoint file.
            
        Returns
        -------
        save_path : string
            The file the model checkpoint was saved to.
        """
        save_path = os.path.join(self.save_dir, f'{self.save_prefix}.pth.tar')
        
        # Save model on cpu
        torch.save({
            'epoch': self.epoch + 1,
            'valid_acc': valid_acc,
            'best_valid_acc': self.best_valid_acc,
            'best_epoch': self.best_epoch,
            'model': self.model.cpu().state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None
        }, save_path)
        self.model.to(self.device)
        
        if is_best:
            shutil.copyfile(save_path, os.path.join(self.save_dir, 'best.pth.tar'))

        return save_path
    