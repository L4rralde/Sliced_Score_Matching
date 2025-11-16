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

    def sample(self, n_samples: int=1, T: int=100, delta:float=1e-2) -> torch.Tensor:
        with torch.no_grad():
            x = 2*(torch.rand(n_samples, self.n) - 0.5)
            for _ in range(T):
                x = x + delta*self.net(x) + torch.randn_like(x)*(2*delta)**0.5
        
        return x


def sliced_score_estimation(
    score_net: Callable,
    samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #creates a copy to set requires_grad.
    dup_samples = samples.clone()
    dup_samples.requires_grad_(True)

    #Directions
    #B x d. B vectors drawn from N(0, I_{d}).
    vectors = torch.randn_like(dup_samples)
    #Normalize directions.

    #Gradients computation
    #score_net(dup_samples) = s_m. B x d
    score = score_net(dup_samples)
    #(v^T s_m).sum(). Autograd.grad requires a scalar. So, sum() is used
    gradv = torch.sum(score * vectors)
    # v^T \nabla s_m = grad(v^T s_m) = auograd.grad(sum(v^T s_m)). # B x d
    grad2 = torch.autograd.grad(gradv, dup_samples, create_graph=True)[0]

    #Loss computation
    # 1/2 ||s_m||^2. For a mini batch, this is in R^{B}
    loss1 = torch.sum(score * score, dim=-1) / 2.
    #v^T \nabla s_m v. Recall, for a sample, v^T \nabla s_m in R^d.
    #   So v^T \nabla s_m v is scalar. 
    #   Hence, considering batches, this is in R^{B}
    loss2 = torch.sum(vectors * grad2, dim=-1)
    loss = loss1 + loss2 #R^B

    #Tuple of 3 tensors of shape empty shape, i.e, scalars
    return loss.mean(), loss1.mean(), loss2.mean()


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
            x = x + torch.randn_like(x)
            loss, *_ = sliced_score_estimation(score_net, x)

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

