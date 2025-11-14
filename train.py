import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import ssm


class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    train_set, val_set, _ = ssm.make_datasets()
    train_dataloader = DataLoader(
        train_set,
        batch_size = 32,
        shuffle = True
    )
    val_dataloader = DataLoader(val_set, batch_size=128)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = ssm.SlicedScoreMatching(ScoreNet(), m=2).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    model, training_stats = ssm.train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=10,
        device=device,
        verbose=True
    )


if __name__ == '__main__':
    main()
