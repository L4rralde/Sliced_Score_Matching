
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def make_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    full_train_set = MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )
    train_set, val_set = torch.utils.data.random_split(
        full_train_set,
        [0.8, 0.2],
        generator = torch.Generator().manual_seed(123)
    )
    test_set = MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    return train_set, val_set, test_set
