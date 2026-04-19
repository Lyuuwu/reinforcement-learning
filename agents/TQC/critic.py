import torch
import torch.nn as nn

from shared.mlp import MLPHead

class Critic(nn.Module):
    '''
    ref: TQC page 5.
    
    For all MuJoCo experiments, 
    we use N = 5 critic networks with three hidden layers of 512 neurons each,
    M = 25 atoms, and the best number of dropped atoms per network d ∈ [0..5], if not stated otherwise.
    '''
    
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, atom_num: int):
        super().__init__()
        
        self.head = nn.Sequential(
            MLPHead(obs_dim + act_dim, hidden),
            MLPHead(hidden, hidden),
            MLPHead(hidden, hidden),
            nn.Linear(hidden, atom_num)
        )
        
    def forward(self, obs, act):
        s_a = torch.cat([obs, act], dim=-1)
        return self.head(s_a)
