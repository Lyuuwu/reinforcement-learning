import os
import time
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import AgentBase, BaseConfig

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

@dataclass
class TrainerConfig(BaseConfig):
    # --- schedule ---
    total_env_steps: int = 1_000_000
    warmup_steps: int = 5_000   # only for off-policy (prefill)

    # --- eval / log / save ---
    eval_interval: int = 10_000
    eval_episodes: int = 10
    log_interval: int = 5000
    save_interval: int = 100_000

    # --- misc ---
    save_dir: str = 'runs/default'
    save_buffer: bool = False
    save_checkpoint: bool = False
    resume: str | None = None
    seed: int = 0

class TrainerBase(ABC):
    def __init__(
        self,
        agent: AgentBase,
        vec_env,
        eval_env,
        logger,
        config: TrainerConfig,
        device: torch.device
    ):
        self.agent = agent
        self.vec_env = vec_env
        self.eval_env = eval_env
        self.logger = logger
        self.config = config
        self.device = device
        
        self.num_envs = vec_env.num_envs
        self.global_env_step = 0
        self.global_update_step = 0
        self._start_time = time.time()
    
    def run(self) -> None:
        self._setup()
        try:
            self._main_loop()
        finally:
            self.close()
    
    @abstractmethod
    def _main_loop(self) -> None: ...
    
    def _setup(self) -> None:
        seed_everything(self.config.seed)
        print(f'[SetUp] seed={self.config.seed}')
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        if self.config.resume is not None:
            self.load_checkpoint(self.config.resume)
            print(f'[Trainer] Resumed from {self.config.resume}, '
                  f'step={self.global_env_step}')
        else:
            print('[Trainer] Starting fresh training')
        
        self.agent.train()
 
    def evaluate(self) -> dict[str, float]:
        self.agent.eval()
        returns = []
        
        for ep in range(self.config.eval_episodes):
            obs, _ = self.eval_env.reset(seed=self.config.seed + ep)
            done, ep_ret = False, 0.0
            
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.agent.act(obs_t).squeeze(0).cpu().numpy()
                obs, r, term, trun, _ = self.eval_env.step(action)
                ep_ret += r
                done = term or trun
            
            returns.append(ep_ret)
        
        self.agent.train()
        
        mean, std = float(np.mean(returns)), float(np.std(returns))
        self.logger.eval_log(self.global_env_step, mean)
        return {'eval/return_mean': mean, 'eval/return_std': std}

    def close(self) -> None:
        self.save_checkpoint(tag='final', include_buffer=self.config.save_buffer)
        self.logger.close()

    # --- checkpoint ---
    def save_checkpoint(self, tag: str='latest', include_buffer: bool=False) -> Path | None:
        if not self.config.save_checkpoint:
            return None
        path = Path(self.config.save_dir) / f'ckpt_{tag}.pt'
        state = {
            'agent': self.agent.state_dict(),
            'optimizers': self._get_optim_state(),
            'global_env_step': self.global_env_step,
            'global_update_step': self.global_update_step,
            'config': self.config.to_dict()
        }
        
        if include_buffer:
            state['buffer'] = self._get_buffer_state()
        
        torch.save(state, path)
        
        print(f'[Trainer] Saved checkpoint to {path}')
        
        return path
    
    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(state['agent'])
        self._load_optim_state(state.get('optimizers', {}))
        self.global_env_step = state['global_env_step']
        self.global_update_step = state['global_update_step']
        if 'buffer' in state:
            self._load_buffer_state(state['buffer'])
    
    # --- hook ---
    def _get_optim_state(self) -> dict: return {}
    def _load_optim_state(self, state: dict) -> None: pass
    def _get_buffer_state(self) -> dict: return {}
    def _load_buffer_state(self, state: dict) -> None: pass