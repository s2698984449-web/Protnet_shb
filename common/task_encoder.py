"""Task representation from material / process / defect-history metadata."""

import torch
import torch.nn as nn


class TaskEncoder(nn.Module):
    """Map domain metadata vectors to 64-d task embedding τ."""

    def __init__(self, material_dim=12, process_dim=8, history_dim=10, out_dim=64):
        super().__init__()
        self.material_net = nn.Sequential(
            nn.Linear(material_dim, 64), nn.ReLU(), nn.Linear(64, 32),
        )
        self.process_net = nn.Sequential(
            nn.Linear(process_dim, 32), nn.ReLU(), nn.Linear(32, 16),
        )
        self.history_net = nn.Sequential(
            nn.Linear(history_dim, 32), nn.ReLU(), nn.Linear(32, 16),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, material_params, process_params, defect_history):
        m = self.material_net(material_params)
        p = self.process_net(process_params)
        h = self.history_net(defect_history)
        return self.fusion(torch.cat([m, p, h], dim=-1))


def meta_vectors_to_tensors(meta, device):
    """Convert domain meta dict to batched tensors on device."""
    material = torch.tensor(meta['material_params'], dtype=torch.float32, device=device)
    process = torch.tensor(meta['process_params'], dtype=torch.float32, device=device)
    history = torch.tensor(meta['defect_history'], dtype=torch.float32, device=device)
    if material.dim() == 1:
        material = material.unsqueeze(0)
    if process.dim() == 1:
        process = process.unsqueeze(0)
    if history.dim() == 1:
        history = history.unsqueeze(0)
    return material, process, history
