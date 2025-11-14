import torch
from tqdm import tqdm
import numpy as np


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device=torch.device('cpu')
) -> dict:
    
    num_train_samples = len(train_dataloader.dataset)
    train_losses = []

    for _ in tqdm(range(1, epochs+1)):
        model.train()
        train_loss = 0.0

        for x, _ in train_dataloader:
            optimizer.zero_grad()

            batch_size = x.size(0)
            x = x.view(batch_size, -1).to(device)
            loss = model(x)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item() * batch_size
        train_loss /= num_train_samples

        train_losses.append(train_loss)
        tqdm.write(f"Train loss: {train_loss}")
        

    return model, train_losses

