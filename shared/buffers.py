from abc import ABC, abstractmethod

import numpy as np
import torch

# ===================
# Base Buffer
# ===================
class BufferBase(ABC):
    @abstractmethod
    def push(self, obs, action, reward, next_obs, done) -> None: ...
    
    @abstractmethod
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]: ...
    
    @abstractmethod
    def __len__(self) -> int: ...
    
    @abstractmethod
    def state_dict(self) -> dict: ...
    
    @abstractmethod
    def load_state_dict(self, state: dict) -> None: ...

# ===================
# Buffers
# ===================
class CPUReplayBuffer(BufferBase):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        batch_size: int,
        device: torch.device
    ):
        self.cap = capacity
        self.device = device
        self._ptr = 0
        self._size = 0
        
        # --- contents ---
        mk = lambda shape: np.empty(shape, dtype=np.float32)
        self.obs        = mk((capacity, obs_dim))
        self.action     = mk((capacity, action_dim))
        self.reward     = mk((capacity, 1))
        self.next_obs   = mk((capacity, obs_dim))
        self.not_done   = mk((capacity, 1))
        
        # --- batch ---
        self._batch = {
            'obs':      np.empty((batch_size, obs_dim),    dtype=np.float32),
            'action':   np.empty((batch_size, action_dim), dtype=np.float32),
            'reward':   np.empty((batch_size, 1),          dtype=np.float32),
            'next_obs': np.empty((batch_size, obs_dim),    dtype=np.float32),
            'not_done': np.empty((batch_size, 1),          dtype=np.float32),
        }
        
    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        p = self._ptr
        self.obs[p] = obs
        self.action[p] = action
        self.reward[p] = reward
        self.next_obs[p] = next_obs
        self.not_done[p] = 1.0 - float(done)
        
        self._ptr = (p + 1) % self.cap
        self._size = min(self._size + 1, self.cap)
        
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._size, size=batch_size)
        
        np.take(self.obs,      idx, axis=0, out=self._batch['obs'])
        np.take(self.action,   idx, axis=0, out=self._batch['action'])
        np.take(self.reward,   idx, axis=0, out=self._batch['reward'])
        np.take(self.next_obs, idx, axis=0, out=self._batch['next_obs'])
        np.take(self.not_done, idx, axis=0, out=self._batch['not_done'])
        
        return {k: self._to_tensor(v) for k, v in self._batch.items()}
    
    def __len__(self) -> int:
        return self._size
    
    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).to(self.device, non_blocking=True)
    
    def state_dict(self):
        n = self._size
        return {
            'obs':      self.obs[:n].copy(),
            'action':   self.action[:n].copy(),
            'reward':   self.reward[:n].copy(),
            'next_obs': self.next_obs[:n].copy(),
            'not_done': self.not_done[:n].copy(),
            'ptr': self._ptr, 'size': self._size,
        }
    
    def load_state_dict(self, state):
        n = state['size']
        self.obs[:n]      = state['obs']
        self.action[:n]   = state['action']
        self.reward[:n]   = state['reward']
        self.next_obs[:n] = state['next_obs']
        self.not_done[:n] = state['not_done']
        self._ptr, self._size = state['ptr'], state['size']
        
class GPUReplayBuffer(BufferBase):   
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        batch_size: int,
        device: torch.device
    ):
        self.cap = capacity
        self.device = device
        self._ptr = 0
        self._size = 0
        
        # --- content ---
        mk = lambda shape: torch.empty(shape, dtype=torch.float32, device=device)
        self.obs      = mk((capacity, obs_dim))
        self.action   = mk((capacity, action_dim))
        self.reward   = mk((capacity, 1))
        self.next_obs = mk((capacity, obs_dim))
        self.not_done = mk((capacity, 1))
    
    def push(self, obs, action, reward, next_obs, done):
        p = self._ptr
        self.obs[p].copy_(self._to_tensor(obs))
        self.action[p].copy_(self._to_tensor(action))
        self.next_obs[p].copy_(self._to_tensor(next_obs))
        self.reward[p, 0]   = float(reward)
        self.not_done[p, 0] = 1.0 - float(done)
        
        self._ptr  = (p + 1) % self.cap
        self._size = min(self._size + 1, self.cap)
    
    def sample(self, batch_size):
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return {
            'obs':      self.obs.index_select(0, idx),
            'action':   self.action.index_select(0, idx),
            'reward':   self.reward.index_select(0, idx),
            'next_obs': self.next_obs.index_select(0, idx),
            'not_done': self.not_done.index_select(0, idx),
        }
    
    def __len__(self):
        return self._size
    
    def _to_tensor(self, x) -> torch.Tensor:
        return torch.from_numpy(np.asarray(x)).to(self.device, non_blocking=True)
    
    def state_dict(self):
        n = self._size

        return {
            'obs':      self.obs[:n].cpu(),
            'action':   self.action[:n].cpu(),
            'reward':   self.reward[:n].cpu(),
            'next_obs': self.next_obs[:n].cpu(),
            'not_done': self.not_done[:n].cpu(),
            'ptr': self._ptr, 'size': self._size,
        }
    
    def load_state_dict(self, state):
        n = state['size']
        self.obs[:n]      = state['obs'].to(self.device)
        self.action[:n]   = state['action'].to(self.device)
        self.reward[:n]   = state['reward'].to(self.device)
        self.next_obs[:n] = state['next_obs'].to(self.device)
        self.not_done[:n] = state['not_done'].to(self.device)
        self._ptr, self._size = state['ptr'], state['size']
        
# ======================
# Build Buffer
# ======================

def build_buffer(kind:str,
                 obs_dim: int,
                 action_dim: int,
                 capacity: int,
                 batch_size: int,
                 device) -> BufferBase:
    
    kind = kind.lower()
    if kind == 'cpu':
        buffer = CPUReplayBuffer
    elif kind == 'gpu':
        if device.type != 'cuda':
            raise ValueError('GPUReplayBuffer requires a cuda device')
        buffer = GPUReplayBuffer
    else:
        raise ValueError(f'Unknown buffer kind: {kind!r}')
    
    return buffer(obs_dim, action_dim, capacity, batch_size, device)
