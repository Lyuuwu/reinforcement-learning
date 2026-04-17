import copy

import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, beta: float):
        self.beta = beta
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        
    @torch.no_grad()
    def update(self, model: nn.Module):
        for param, ema_param in zip(model.parameters(), self.ema_model.parameters()):
            ema_param.lerp_(param, 1 - self.beta)
            
    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)