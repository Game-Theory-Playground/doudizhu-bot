
# irl/reward_net.py

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim=49*54, hidden_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        x = torch.cat([state.view(state.size(0), -1), action.unsqueeze(1).float()], dim=1)
        return self.model(x)

    def reward(self, state, action):
        with torch.no_grad():
            d = self.forward(state.unsqueeze(0), action.unsqueeze(0))
            return -torch.log(1 - d + 1e-8)