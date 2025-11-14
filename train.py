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

    model = ssm.SlicedScoreMatching(ssm.ScoreNet(784)).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    model, training_stats = ssm.train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        epochs=100,
        device=device,
    )


if __name__ == '__main__':
    main()
