"""策略网络。由 scripts/train_cspd.py 在每个 episode 里调用，没有单独运行命令。

输入：每一类一行，只来自该类参考图，不看待检图，也不看类别名字。
  零件码 128：这一集表面长什么样
  该类外观 128：这类参考图的特征均值
  该类工况 D：这类参考图的工况槽位均值（缺测维为 0）
  离散度 1：这类几张参考图差多少
  特征差 1：原图特征和工况调制后的特征差多少

输出：每一类一行，交给下游距离计算，不是类别分数。
  g：信工况调制特征的比例，0.5 表示和原图各半
  alpha：细尺度、粗尺度的权重，两类加起来为 1
  gate：每个通道的敏感系数，1 表示不偏袒任何通道
  tau：距离放大倍数，1 表示不放大也不缩小

学习：待检图分类损失反传到这些输出上。参考图长得像细线时，损失会把细尺度权重推高。
起始：最后一层为 0，所以一开始各类都是 g=0.5、尺度均分、通道全 1、tau=1。
"""

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, state_dim, feat_dim=128, n_scales=2, rho=0.5, hidden=128):
        super().__init__()
        self.rho = float(rho)
        self.n_scales = n_scales
        self.feat_dim = feat_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.g_head = nn.Linear(hidden, 1)
        self.alpha_head = nn.Linear(hidden, n_scales)
        self.gate_head = nn.Linear(hidden, feat_dim)
        self.tau_head = nn.Linear(hidden, 1)
        for head in (self.g_head, self.alpha_head, self.gate_head, self.tau_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, state):
        """state: (state_dim,) 一类，或 (类数, state_dim)。"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.trunk(state)
        g = torch.sigmoid(self.g_head(h)).view(-1)
        alpha = torch.softmax(self.alpha_head(h), dim=-1)
        gate = 1.0 + self.rho * torch.tanh(self.gate_head(h))
        tau = torch.exp(self.tau_head(h).clamp(-2.0, 2.0)).view(-1)
        return g, alpha, gate, tau


class EpisodeBetaPolicy(nn.Module):
    """课程学习初版：整集一个 β。最后一层零初始化 → β 从 0.25 起步。"""

    def __init__(self, state_dim, hidden=(64, 32), beta_center=0.25, beta_scale=0.25):
        super().__init__()
        self.beta_center = float(beta_center)
        self.beta_scale = float(beta_scale)
        layers = []
        dim = int(state_dim)
        for width in hidden:
            layers.extend([nn.Linear(dim, int(width)), nn.ReLU()])
            dim = int(width)
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        u = self.head(self.trunk(state)).view(-1)
        beta = self.beta_center + self.beta_scale * torch.tanh(u)
        return beta.view(-1)
