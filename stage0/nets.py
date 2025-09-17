import torch
import torch.nn as nn


class Stage0LogNet(nn.Module):
    def __init__(self, input_dim, hidden=64, min_sigma=1e-6):
        super().__init__()
        self.min_sigma = min_sigma
        self.f = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.mu = nn.Linear(hidden, 1)
        self.ls = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.f(x)
        mu = self.mu(h)
        sigma = torch.exp(self.ls(h)) + self.min_sigma
        return mu, sigma  # σ is annualized

