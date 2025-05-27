import torch
import torch.nn as nn

class MapperMLP(nn.Module):
    """
    A simple MLP that maps a CLAP embedding (size 512) to a target vector of audio parameters.
    You can change output_dim depending on how many parameters you need (e.g. f0 + amp + z).
    """
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super(MapperMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
