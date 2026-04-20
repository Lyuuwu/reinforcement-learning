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

_CRITICAL_KEYS = {'seed', 'batch_size', 'updates_per_step'}

def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)
    
def _get_rng_state() -> dict:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
    }
    
def _set_rng_state(s: dict) -> None:
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    if s.get('torch_cuda') is not None:
        torch.cuda.set_rng_state_all(s['torch_cuda'])

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
        self._resumed = False
    
    def run(self) -> None:
        self._setup()
        try:
            self._main_loop()
        finally:
            self._finalize()
    
    @abstractmethod
    def _main_loop(self) -> None: ...
    
    # --- SetUP ---
    def _setup(self) -> None:
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt_path = self._resolve_resume_path()
        
        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)
            self._resumed = True
            print(f'[Trainer] Resumed from {ckpt_path} @ sstep {self.global_env_step}')
        else:
            seed_everything(self.config.seed)
            print(f'[Trainer] Fresh start, seed={self.config.seed}')
        
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

    def _finalize(self) -> None:
        self._save_checkpoint(tag='final', include_buffer=self.config.save_buffer)
        self.logger.close()

    def _resolve_resume_path(self) -> Path | None:
        if self.config.resume is None:
            return None
        
        if self.config.resume == 'auto':
            p = Path(self.config.save_dir) / 'ckpt_latest.pt'
            return p if p.exists() else None
        
        if self.config.resume:
            p = Path(self.config.resume)
            if not p.exists():
                raise FileNotFoundError(f'--resume path not found: {p}')
        
        return None

    # --- checkpoint ---
    def _save_checkpoint(self, tag: str = 'latest',
                         include_buffer: bool = False) -> Path | None:
        if not self.config.save_checkpoint:
            return None
        
        save_dir = Path(self.config.save_dir)
        ckpt_path = save_dir / f'ckpt_{tag}.pt'

        state = {
            'meta': {
                'tag': tag,
                'saved_at': __import__('datetime').datetime.now().isoformat(),
            },
            'agent': self.agent.state_dict(),
            'optimizers': self._get_optim_state(),
            'global_env_step': self.global_env_step,
            'global_update_step': self.global_update_step,
            'config': self.config.to_dict(),
            'rng': _get_rng_state(),
        }
        _atomic_torch_save(state, ckpt_path)

        if include_buffer:
            buf_path = save_dir / f'buffer_{tag}.pt'
            _atomic_torch_save(self._get_buffer_state(), buf_path)

        print(f'[Trainer] Saved {ckpt_path.name}'
              + (f' (+ buffer)' if include_buffer else ''))
        return ckpt_path

    def _load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self._validate_config(state['config'])
        self.agent.load_state_dict(state['agent'])
        self._load_optim_state(state.get('optimizers', {}))
        self.global_env_step = state['global_env_step']
        self.global_update_step = state['global_update_step']
        if state.get('rng') is not None:
            _set_rng_state(state['rng'])

        buf_path = path.parent / f'buffer_{path.stem.split("_", 1)[1]}.pt'
        if buf_path.exists():
            self._load_buffer_state(torch.load(buf_path))
            print(f'[Trainer] Loaded buffer from {buf_path.name}')
        else:
            print(f'[Trainer] No buffer found, will re-prefill')

    def _validate_config(self, saved: dict) -> None:
        current = self.config.to_dict()
        mismatches = [k for k in _CRITICAL_KEYS
                      if k in saved and k in current and saved[k] != current[k]]
        if mismatches:
            lines = [f'  {k}: saved={saved[k]!r} vs current={current[k]!r}'
                     for k in mismatches]
            raise ValueError(
                'Critical config mismatch on resume:\n' + '\n'.join(lines)
            )
    
    # --- hook ---
    def _get_optim_state(self) -> dict: return {}
    def _load_optim_state(self, state: dict) -> None: pass
    def _get_buffer_state(self) -> dict: return {}
    def _load_buffer_state(self, state: dict) -> None: pass