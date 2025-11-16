import os

import torch
import torchvision.transforms as transforms
from ssm.vae_ssm import(
    MLPImplicitEncoder,MLPDecoder, MLPLatentScore,
    Config,
    train
)
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    config = Config({
        'batch_size': 128,
        'n_epochs': 200,
        'n_iters': 100000,
        'z_dim': 32,
        'eps_dim': 32,
        'lr': 1e-3
    })

    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
    ])

    dataset = MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    imp_encoder = MLPImplicitEncoder(config).to(device)
    decoder = MLPDecoder(config).to(device)
    score = MLPLatentScore(config).to(device)
    
    try:
        train(imp_encoder, decoder, score, dataloader, config, device)
    except Exception as e:
        print(e)
    finally:
        os.makedirs('models', exist_ok=True)
        torch.save(imp_encoder.state_dict(), 'models/encoder.pth')
        torch.save(decoder.state_dict(), 'models/decoder.pth')
        torch.save(score.state_dict(), 'models/latent_score.pth')

if __name__ == '__main__':
    main()
