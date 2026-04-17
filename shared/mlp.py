import torch
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        '''
        linear + relu
        '''
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.head(x)