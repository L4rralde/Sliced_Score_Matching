from typing import Tuple, Callable

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


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


class ScoreNet(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.net = nn.Sequential(
            nn.Linear(n, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x Tensor B x n
        return self.net(x) # B x n

    def sample(self, n_samples: int=1, T: int=10) -> torch.Tensor:
        delta = 1/T
        with torch.no_grad():
            x = torch.rand(n_samples, self.n)
            for _ in range(T):
                x = x + delta*self.net(x) + torch.randn_like(x)*(2*delta)**0.5
        
        return x


def sliced_score_estimation(
    x: torch.Tensor,
    score_net: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #x = x.clone().detach().requires_grad_(True) #to detach or to not?
    x = x.clone().requires_grad_(True)

    #Random directions. 
    vectors = torch.randn_like(x) #B x n
    vectors = nn.functional.normalize(vectors, dim=-1) # B x n

    #s_m(x; theta)
    score = score_net(x) #B x n

    #Projections. v^Ts_m(x; theta)
    projections = (vectors * score).sum(dim=-1) #B, v^T s
    #v^T \nabla s_m(x; theta) = grad(v^Ts_m(x, theta), x)
    grad2 = torch.autograd.grad(projections.sum(), x, create_graph=True)[0] #B x n

    #J = 1/2 ||score||_2^2
    loss_1 = 0.5 * (score * score).sum(dim=-1) #B
    #J_2 = v^T \nabla s_m(x; theta) v
    loss2 = torch.sum(vectors * grad2, dim=-1) #B

    # J = J + v^T \nabla s_m(x; theta) v
    loss = loss_1 + loss2

    return loss.mean(), loss_1.mean(), loss2.mean() 


def train(
    score_net: nn.Module,
    sliced_score_estimation: Callable,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device=torch.device('cpu')
) -> dict:
    
    num_train_samples = len(train_dataloader.dataset)
    train_losses = []
    best_loss = 0

    for _ in tqdm(range(1, epochs+1)):
        score_net.train()
        train_loss = 0.0

        for x, _ in train_dataloader:
            optimizer.zero_grad()

            batch_size = x.size(0)
            x = x.view(batch_size, -1).to(device)
            x = x + 0.1*torch.randn_like(x)
            loss, *_ = sliced_score_estimation(x, score_net)

            loss.backward()
            optimizer.step()

            train_loss += loss.data.item() * batch_size
        train_loss /= num_train_samples

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(score_net.state_dict(), 'models/scorenet.pth')

        train_losses.append(train_loss)
        tqdm.write(f"Train loss: {train_loss}")
        

    return train_losses

