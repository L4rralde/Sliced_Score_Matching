import torch
from tqdm import tqdm
import numpy as np


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device=torch.device('cpu'),
    verbose: bool=False,
) -> dict:
    
    num_train_samples = len(train_dataloader.dataset)
    num_val_samples = len(val_dataloader.dataset)
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        train_loss = 0.0

        for x, y in train_dataloader:
            optimizer.zero_grad()

            batch_size = x.size(0)
            x = x.view(batch_size, -1).to(device)
            score, loss = model(x)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item() * batch_size
        train_loss /= num_train_samples

        model.eval()
        """
        val_loss = 0.0

        for x, y in val_dataloader:
            batch_size = x.size(0)
            x = x.view(batch_size, -1).to(device)
            score, loss = model(x)

            val_loss += loss.data.item() * batch_size
        
        val_loss /= num_val_samples
        """

        if verbose:
            tqdm.write(f"epoch: {epoch}")
            tqdm.write(f"Train loss: {train_loss: .4f}")
            #tqdm.write(f"Vall loss: {val_loss: .4f}")
        
        train_losses.append(train_loss)
        #val_losses.append(val_losses)
    
    training_stats = {
        'train_loss': np.array(train_losses),
        'test_loss': np.array(val_losses)
    }

    return model, training_stats

