import numpy as np
import torch

class ReplayBuffer:
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
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.action = np.empty((capacity, action_dim), dtype=np.float32)
        self.reward = np.empty((capacity, 1), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.not_done = np.empty((capacity, 1), dtype=np.float32)
        
        # --- batch ---
        self._batch = {
            'obs':      np.empty((batch_size, obs_dim),    dtype=np.float32),
            'action':   np.empty((batch_size, action_dim), dtype=np.float32),
            'reward':   np.empty((batch_size, 1),          dtype=np.float32),
            'next_obs': np.empty((batch_size, obs_dim),    dtype=np.float32),
            'not_done': np.empty((batch_size, 1),          dtype=np.float32),
        }
        self._idx_buf = np.empty(batch_size, dtype=np.int64)
        
    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        self.obs[self._ptr] = obs
        self.action[self._ptr] = action
        self.reward[self._ptr] = reward
        self.next_obs[self._ptr] = next_obs
        self.not_done[self._ptr] = 1.0 - float(done)
        
        self._ptr = (self._ptr + 1) % self.cap
        self._size = min(self._size + 1, self.cap)
        
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        np.random.randint(0, self._size, size=batch_size, out=self._idx_buf)
        idx = self._idx_buf
        
        np.take(self.obs,      idx, axis=0, out=self._batch['obs'])
        np.take(self.action,   idx, axis=0, out=self._batch['action'])
        np.take(self.reward,   idx, axis=0, out=self._batch['reward'])
        np.take(self.next_obs, idx, axis=0, out=self._batch['next_obs'])
        np.take(self.not_done, idx, axis=0, out=self._batch['not_done'])
        
        return {k: self._to_tensor(v) for k, v in self._batch.items()}
        
    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).to(self.device, non_blocking=True)
    
    def __len__(self) -> int:
        return self._size