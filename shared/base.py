import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
@dataclass
class BaseConfig:
    def to_dict(self) -> dict:
        return asdict(self)

    def get(self, key: str, default=None):
        return getattr(self, key, default)
    
    def override(self, **kwargs) -> 'BaseConfig':
        c = copy.deepcopy(self)
        for k, v in kwargs.items():
            if not hasattr(c, k):
                raise ValueError(f'Unknown config key: {k}')
            setattr(c, k, v)
        return c
    
class AgentBase(nn.Module, ABC):
    @abstractmethod
    def sample(self, obs):
        ''' stochastic for train '''
    
    @abstractmethod
    def act(self, obs):
        ''' deterministic for eval '''
    
    @abstractmethod
    def update(self, batch) -> dict:
        ''' one gradient step '''
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))