import functools
import itertools
from typing import Callable, Any, Tuple
import os

import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class Config:
    def __init__(self, config: dict={}) -> None:
        self.config = config

    def __getattr__(self, __name: str) -> Any:
        return self.config[__name]


class MLPDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.z_dim = config.z_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 784)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.view(inputs.shape[0], -1) #B x z_dim
        logits = self.mlp(flattened) #B x 784
        logits = logits.view(inputs.shape[0], 1, 28, 28) #B x 1 x 28 x 28
        return logits

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(n_samples, self.z_dim) #B x z_dim
            logits = self.mlp(z)
            samples = torch.sigmoid(logits)
            samples = samples.view(n_samples, 1, 28, 28)

        return samples


class MLPScore(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.z_dim = config.z_dim
        self.mlp = nn.Sequential(
            nn.Linear(784 + self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.z_dim)
        )

    def forward(self, X: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # X in {0.0,1.0}^{B x 28 x 28}
        #z must be B x z_dim
        X = X.view(X.shape[0], -1) #Image to B x n, e.g., B x 784
        Xz = torch.cat([X, z], dim=-1) #B x (n + z_dim). [X, z] per sample
        h = self.mlp(Xz) #B x z_dim
        return h

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        raise NotImplementedError("Requires X as input :(")

class MLPLatentScore(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.z_dim = config.z_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.z_dim)
        )

    def forward(self, X: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        #z must be B x z_dim
        h = self.mlp(z) #B x z_dim
        return h

    def sample(self, n_samples: int=1, T: int=100) -> torch.Tensor:
        delta = 1/T
        with torch.no_grad():
            z = torch.randn(n_samples, self.z_dim)
            for _ in range(T):
                z = z + delta*self.mlp(z) + torch.randn_like(z)*(2*delta)**0.5
        return z


class MLPImplicitEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.eps_dim = config.eps_dim #eps is added noise
        self.z_dim = config.z_dim

        self.main = nn.Sequential(
            nn.Linear(784 + self.eps_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.z_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        eps = torch.randn(batch_size, self.eps_dim, device=x.device) #B random vectors drawn fom N(0, I_d) d=eps_dim
        flattened_x = x.view(batch_size, -1) #B x n, e.g, B*784
        x_eps = torch.cat([flattened_x, eps], dim=-1) #B x (n + eps_dim). B vectors of [X, eps]
        z = self.main(x_eps) #B x z_dim. B latent vectors
        return z


def elbo_ssm(
    imp_encoder: MLPImplicitEncoder,
    decoder: MLPDecoder,
    score: MLPScore,
    score_opt: torch.optim.Optimizer,
    X: torch.Tensor,
    training: bool=True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #1. Compute ssm_loss
    #X in B x 28 x 28
    dup_X = X.unsqueeze(0).expand(1, *X.shape).contiguous().view(-1, *X.shape[1:])
    #dup_X = X.clone()
    #dup_X in B x 28 x 28. Same shape. However, it is another object.
    z = imp_encoder(X) #B x z_dim
    ssm_loss, *_ = sliced_score_estimation_vr(functools.partial(score, dup_X), z)
    #ssm_loss in R

    # 1.1 Updates score_net with backpropagation
    if training:
        score_opt.zero_grad()
        ssm_loss.backward()
        score_opt.step()

    #2. Compute vae loss: elbo
    z = imp_encoder(X) #B x z_dim Graph independent to the one used for score_net update

    #Reconstruction loss
    x_logits = decoder(z) #X in R^{B x 784}
    recon = nn.functional.binary_cross_entropy_with_logits(
        input=x_logits,
        target=X,
        reduction='sum'
    )
    recon /= x_logits.shape[0] #Mean (mini-batch) reconstruction loss

    nlogpz = z ** 2 / 2. + np.log(2. * np.pi) / 2.
    nlogpz = nlogpz.sum(dim=-1) # A sort of regularization?

    scores = score(X, z)
    entropy_loss = (scores.detach() * z).sum(dim=-1) #Well, this is not Kullback-Leibler, is it?

    loss = recon + nlogpz + entropy_loss

    loss = loss.mean()

    return loss, ssm_loss, recon


def sliced_score_estimation_vr(
    score_net: Callable,
    samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #samples in B x z_dim. samples are latent vectors, i.e., samples in Z
    dup_samples = samples.unsqueeze(0).expand(1, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    #dup_samples = samples.clone()
    dup_samples.requires_grad_(True) #B x z_dim. Same shape as samples. dup_samples has the same data, but is another object.
    vectors = torch.randn_like(dup_samples) #B x z_dim. B vectors drawn from N(0, I_{z_dim}).
    #vectors = torch.nn.functional.normalize(vectors, dim=-1)

    #score_net already includes X (the image), but as a constant. refer to functools.partial(score_net, X)
    #Then, the following runs score_net.forward(X, dup_samples).
    grad1 = score_net(dup_samples) #score_net(X, dup_samples), s_m
    gradv = torch.sum(grad1 * vectors) #(v^T s_m).sum(). Autograd.grad requires a scalar. So, sum() is used
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2. # 1/2 ||s_m||^2. For a mini batch, this is in R^{B}
    grad2 = torch.autograd.grad(gradv, dup_samples, create_graph=True)[0] # v^T \nabla s_m = grad(v^T s_m) = auograd.grad(sum(v^T s_m)). # B x z_dim
    loss2 = torch.sum(vectors * grad2, dim=-1) #v^T \nabla s_m v. Recall, for a sample, v^T \nabla s_m in R^z_dim. So v^T \nabla s_m v is scalar. Hence, considering batches, this is in R^{B}

    loss = loss1 + loss2 #R^B
    return loss.mean(), loss1.mean(), loss2.mean() #Tuple of 3 tensors of shape empty shape, i.e, scalars


def train(
    imp_encoder: MLPImplicitEncoder,
    decoder: MLPDecoder,
    score: MLPScore,
    dataloader: torch.utils.data.DataLoader,
    config: Config,
    device: torch.device
):
    opt_vae = torch.optim.RMSprop(
        itertools.chain(imp_encoder.parameters(), decoder.parameters()),
        lr=config.lr
    )

    opt_score = torch.optim.RMSprop(
        score.parameters(),
        config.lr
    )

    step = 0
    for epoch in tqdm(range(1, config.n_epochs)):
        for X, y in dataloader:
            #if len(X.shape) == 4:
            #    X = X.squeeze(1)
            X = X.to(device)
            #Random (uniform) Binarization:
            X = (torch.rand_like(X) <= X).float() # B x 28 x 28

            decoder.train()
            imp_encoder.train()
            loss, ssm_loss, *_ = elbo_ssm(
                imp_encoder,
                decoder,
                score,
                opt_score,
                X,
                training=True,
            )

            if step % 10 == 0:
                tqdm.write(f"Epoch: {epoch}, step: {step}. ssm_loss: {ssm_loss: .4f}, loss: {loss: .4f}")

            opt_vae.zero_grad()
            loss.backward()
            opt_vae.step()

            step += 1
            if step >= config.n_iters:
                return

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
        if isinstance(score, MLPLatentScore):
            torch.save(score.state_dict(), 'models/latent_score.pth')
        else:
            torch.save(score.state_dict(), 'models/score.pth')


if __name__ == '__main__':
    main()
