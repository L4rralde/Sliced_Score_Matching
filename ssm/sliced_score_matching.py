import torch
import torch.nn.functional as F


class ScoreNet(torch.nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x Tensor B x n
        return self.net(x) # B x n


class SlicedScoreMatching(torch.nn.Module):
    def __init__(self,
        score_net: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.score_net = score_net

    def forward(self, x: torch.Tensor) -> tuple:
        x = x.clone().detach().requires_grad_(True)

        #s_m(x; theta)
        score = self.score_net(x) #B x n
        
        #Random directions. 
        vectors = torch.randn_like(x) #B x n
        vectors = F.normalize(vectors, dim=-1) # B x n

        #Projections. v^Ts_m(x; theta)
        projections = (vectors * score).sum(dim=-1) #B, v^T s
        #v^T \nabla s_m(x; theta) = grad(v^Ts_m(x, theta), x)
        grad2 = torch.autograd.grad(projections.sum(), x, create_graph=True)[0] #B x n

        #J = 1/2 ||score||_2^2
        loss_1 = 0.5 * (score * score).sum(dim=-1) #B
        #J_2 = v^T \nabla s_m(x; theta) v
        loss2 = torch.sum(vectors * grad2, dim=-1) #B

        # J = J + v^T \nabla s_m(x; theta) v
        loss = (loss_1 + loss2).mean()

        return loss
