"""连续工况编码、零件编码、FiLM。

零件编码只接收支持集特征。查询图不进入 z_part。
"""

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class FiLM(nn.Module):
    """F' = (1 + clamp(gamma, -rho, rho)) * F + clamp(beta, -rho, rho)。

    gamma / beta 零初始化，起始时等于恒等映射。
    """

    def __init__(self, d_cond=128, d_feat=128, rho=0.5):
        super().__init__()
        self.gamma = nn.Linear(d_cond, d_feat)
        self.beta = nn.Linear(d_cond, d_feat)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        self.rho = float(rho)

    def forward(self, feat, cond_emb):
        # feat: (B, C, H, W), cond_emb: (B, d)
        g = self.gamma(cond_emb).clamp(-self.rho, self.rho)
        b = self.beta(cond_emb).clamp(-self.rho, self.rho)
        g = g.unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + g) + b
