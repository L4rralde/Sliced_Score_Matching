import torch


class SlicedScoreMatching(torch.nn.Module):
    def __init__(self,
        score_net: torch.nn.Module,
        m: int = 1
    ) -> None:
        super().__init__()
        self.score_net = score_net
        self.m = m

    def forward(self, x: torch.Tensor) -> tuple:
        #FIXME
        x.requires_grad_(True)
        score = self.score_net(x) #B x n

        loss_1 = 0.5 * torch.norm(score, dim=-1)**2 # 1/2 ||score||*2. Vector of dimension B.

        batch_size, n = x.shape
        vectors = torch.randn(
            self.m, batch_size, n,
            dtype=x.dtype,
            device=x.device
        )
        vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)
        vectors = vectors.view(self.m, batch_size, n, 1) # M x B x n x 1

        score = score.view(1, batch_size, n, 1) #1 x B x n x 1

        projections = torch.matmul(
            vectors.transpose(-1, -2),
            score
        ) #M x B x 1 x 1

        gradv = torch.autograd.grad(
            projections.sum(),
            x,
            create_graph=True
        )[0] #B x n.
        gradv = gradv.view(1, batch_size, n, 1)

        gradv_v = (vectors*gradv) #M x B x n x 1
        gradv_v = gradv_v.squeeze(-1) # M x B x n
        loss_2 = torch.sum(gradv_v, dim=-1) #M x B
        loss_2 = loss_2.mean(dim=0) #Vector of dimension B.

        loss = (loss_1 + loss_2).mean() #Scalar

        return score.view(batch_size, n), loss
