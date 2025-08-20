# src/mapper/mapper_mlp.py
import torch
import torch.nn as nn

class MLPMapper(nn.Module):
    """Simple MLP that maps text embeddings -> audio embeddings (CLAP space)."""
    def __init__(self, in_dim=512, hidden=(512, 512), out_dim=512, dropout=0.1, norm_out=True):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
        self.norm_out = norm_out

    def forward(self, x):
        y = self.net(x)
        if self.norm_out:
            y = nn.functional.normalize(y, dim=-1, eps=1e-8)
        return y
