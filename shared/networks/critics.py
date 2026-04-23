import torch
import torch.nn as nn

from .mlp import MLPHead

class VCritic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        
        # 2層 hidden layer, 每層 256 個 units
        self. head = nn.Sequential(
            MLPHead(obs_dim, 256),
            MLPHead(256, 256),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.head(x)
    
class QCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        
        self.head = nn.Sequential(
            MLPHead(obs_dim + act_dim, 256),
            MLPHead(256, 256),
            nn.Linear(256, 1)
        )
        
    def forward(self, obs, act):
        s_a = torch.cat([obs, act], dim=-1)
        return self.head(s_a)
