import torch.nn as nn
import torch.optim as optim
import torchsde
import torch

class NeuralSDE(torchsde.SDEIto):
    def __init__(self, dim):
        super().__init__(noise_type = 'diagonal')

        # Drift
        self.f_net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

        # Diffusion network (output >= 0)
        self.g_net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, dim),
            nn.Softplus()
        )

    def f(self, t, y):
        return self.f_net(y)
    
    def g(self, t, y):
        return self.g_net(y)
    
