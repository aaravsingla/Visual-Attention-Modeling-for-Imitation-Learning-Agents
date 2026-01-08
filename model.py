import torch
import torch.nn as nn

class GazePolicy(nn.Module):
    def __init__(self):
        super(GazePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Outputs logits for [Fixate, Move]
        )

    def forward(self, x):
        return self.net(x)
