import torch




def train_model(train_dataset, valid_dataset, test_dataset, net, model_name, batch_size=64, num_epochs=100,
                optimizer=None, scheduler=None, log_file=None, device=torch.device("cpu"),
                seed=None, print_every=1, save_every=10, save_dir='.', resume=None):
    """
    """
    if valid_batch_size is None:
        valid_batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=validbatch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=valid_batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
        # Training
        for notes, chords in train_loader:
            # Transfer to device
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            pass

        # Validation
        with torch.set_grad_enabled(False):
            for notes, chords in valid_loader:
                # Transfer to device
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                pass
            
        # Log printing
        if epoch % print_every == 0:
            pass
        
        # Save checkpoints
        if epoch % save_every == 0:
            pass
    