import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import AgentBase

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class TrainerBase(ABC):
    def __init__(
        self,
        agent: AgentBase,
        vec_env,
        eval_env,
        logger,
        config,
        device: torch.device
    ):
        self.agent = agent
        self.vec_env = vec_env
        self.eval_env = eval_env
        self.logger = logger
        self.config = config
        self.device = device
        
        self.num_envs = vec_env.num_envs
        self._global_env_step = 0
    
    def run(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _main_loop(self) -> None: ...
    
    def _setup(self) -> None:
        resume_path = self.config.get('resume', None)
        if resume_path is not None:
            self._load_checkpoint(resume_path)
            print(f'[Trainer] Resumed from {resume_path}, '
                  f'env_step={self._global_env_step}')
        else:
            print(f'[Trainer] Starting fresh training')
 
    def _load_checkpoint(self, path: str | Path) -> None:
        ...