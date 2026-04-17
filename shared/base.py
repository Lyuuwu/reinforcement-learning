import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

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
    
class BaseAgent:
    @abstractmethod
    def sample(self, obs): ...
    
    @abstractmethod
    def act(self, obs): ...
    
    @abstractmethod
    def set_train(self): ...
    
    @abstractmethod
    def set_eval(self): ...
    
    @abstractmethod
    def train(self, batch): ...