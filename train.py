import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import ssm


def main():
    train_set, *_ = ssm.make_datasets()
    train_dataloader = DataLoader(
        train_set,
        batch_size = 128,
        shuffle = True
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = ssm.ScoreNet(784).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    training_stats = ssm.train(
        score_net=model,
        sliced_score_estimation=ssm.sliced_score_estimation,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        epochs=20,
        device=device,
    )



if __name__ == '__main__':
    main()
