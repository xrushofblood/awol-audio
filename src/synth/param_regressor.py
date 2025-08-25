# src/synth/param_regressor.py
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ParamRegressor(nn.Module):
    """
    Predicts normalized parameters in [0,1]. Use ParamBounds to denormalize.
    """
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        self.mlp = MLP(in_dim, hidden, out_dim, dropout)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        return self.out_act(self.mlp(x))

@dataclass
class ParamBounds:
    names: List[str]
    mins:  torch.Tensor
    maxs:  torch.Tensor

    @staticmethod
    def from_cfg(cfg) -> "ParamBounds":
        ps = cfg["synth"]["params"]
        names = [p["name"] for p in ps]
        mins  = torch.tensor([p["min"] for p in ps], dtype=torch.float32)
        maxs  = torch.tensor([p["max"] for p in ps], dtype=torch.float32)
        return ParamBounds(names, mins, maxs)

    def clamp_denorm(self, y01: torch.Tensor) -> torch.Tensor:
        y = self.mins + y01 * (self.maxs - self.mins)
        return torch.clamp(y, self.mins, self.maxs)

def denorm_to_dict(y01: torch.Tensor, bounds: ParamBounds) -> Dict[str, float]:
    y = bounds.clamp_denorm(y01).detach().cpu().numpy().tolist()
    return {name: float(val) for name, val in zip(bounds.names, y)}
