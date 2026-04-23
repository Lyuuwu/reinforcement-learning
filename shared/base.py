import copy
import collections
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
    def __init__(self):
        super().__init__()
        
        self._pending: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
    
    @abstractmethod
    def sample(self, obs):
        ''' stochastic for train '''
    
    @abstractmethod
    def act(self, obs):
        ''' deterministic for eval '''
    
    @abstractmethod
    def update(self, batch) -> dict:
        ''' one gradient step '''
    
    def _stash(self, name: str, value: torch.Tensor) -> None:
        self._pending[name].append(value.detach())
    
    def flush_metrics(self) -> dict[str, float]:
        if not self._pending:
            return {}
        
        names = list(self._pending.keys())
        means = [torch.stack(self._pending[n]).mean() for n in names]
        packed = torch.stack(means).cpu()
        out = {n: packed[i].item() for i, n in enumerate(names)}
        
        self._pending.clear()
        return out
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))