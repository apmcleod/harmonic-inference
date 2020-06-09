import os

import torch
import torch.nn as nn




class ModelTrainer():
    """
    A ModelTrainer object can be used to train or evaluate a pytorch model.
    """
    
    def __init__(self, model, train_dataset=None, valid_dataset=None, test_dataset=None,
                 device=torch.device("cpu"), seed=None,
                 batch_size=64, valid_batch_size=None, num_epochs=100,
                 optimizer=None, scheduler=None, criterion=None, accuracy=None,
                 log_every=1, log_file=None,
                 save_every=10, save_dir='.', save_prefix='checkpoint'):
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
            
        optimizer : torch.optim
            The optimizer to use during training.
            
        scheduler : torch.scheduler
            The scheduler to use during training (if any).
            
        criterion : torch.criterion
            The loss criterion to use during training and evaluation.
            
        accuracy : func
            A function that takes as input a torch tensor of predictions, followed by ground truth,
            and returns a torch tensor of the accuracy.
            
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
        """
        if valid_batch_size is None:
            valid_batch_size = batch_size
            
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset is not None else None
        self.valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False) if valid_dataset is not None else None
        self.test_loader = DataLoader(test_dataset, batch_size=valid_batch_size, shuffle=False) if test_dataset is not None else None
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.device = device
        self.model = model
        self.model.to(self.device)
        
        self.num_epochs = num_epochs
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.accuracy = accuracy
        
        self.log_every = log_every
        self.log_file = sys.stdout if log_file is None else log_file
        
        self.save_every = save_every
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        
        
        
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
        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_acc = self.iteration(train=True)

            # Validation
            valid_loss, valid_acc = self.iteration(train=False)

            # Log printing
            if epoch % self.log_every == 0:
                self.log([train_loss, train_acc, valid_loss, valid_acc])

            # Save checkpoints
            if epoch % save_every == 0:
                self.save(epoch=epoch)
                
        self.save(epoch=epoch)
    
    
    
    def iteration(train=True):
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
            this_batch_size = len(batch)
            
            # Load data
            notes = [sample['notes'] for sample in batch]
            notes_lengths = [len(note) for note in notes]
            labels = [sample['chords']['one_hot'] for sample in batch]
            
            # Transfer to device
            notes, labels = notes.to(self.device), labels.to(self.device)
            
            if train:
                self.optimizer.zero_grad()
                
            predictions = self.model.forward(notes, notes_length)
            loss = self.criterion(predictions, labels)
            acc = self.accuracy(predictions, labels)
            
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
    
    
    
    def save(self, epoch=None, save_path=None):
        """
        Save the currently loaded model to a file. By default,
        {self.save_dir}/{self.save_prefix}_e{epoch}.model. If save_path is given, the model is saved
        to save_path.
        
        Parameters
        ----------
        epoch : int
            The epoch to write into the file name. If not given, '_e{epoch}' is removed from the filename.
            
        save_path : string
            The file to save the model to, if given.
            
        Returns
        -------
        save_path : string
            The file the model checkpoint was saved to.
        """
        if path is None:
            save_dir = '.' if self.save_dir is None else self.save_dir
            save_prefix = 'checkpoint' if self.save_prefix is None else self.save_prefix
            save_filename = f'{save_prefix}_e{epoch}.model' if epoch is not None else f'{save_prefix}.model'
            save_path = os.path.join(save_dir, save_filename)
        else:
            save_dir = os.path.dirname(save_path)
            if save_dir != '':
                os.makedirs(save_dir, exist_ok=True)
        
        # Save model on cpu
        torch.save(self.model.cpu(), save_path)
        self.model.to(self.device)
        
        print(f"Model saved to {save_path}")
        return save_path
    