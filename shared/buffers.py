import numpy as np
import torch

class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device
    ):
        self.cap = capacity
        self.device = device
        self._ptr = 0
        self._size = 0
        
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.action = np.empty((capacity, action_dim), dtype=np.float32)
        self.reward = np.empty((capacity, 1), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.not_done = np.empty((capacity, 1), dtype=np.float32)
        
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
        idx = np.random.randint(0, self._size, size=batch_size)
        
        return {
            'obs': self._to_tensor(self.obs[idx]),
            'action': self._to_tensor(self.action[idx]),
            'reward': self._to_tensor(self.reward[idx]),
            'next_obs': self._to_tensor(self.next_obs[idx]),
            'not_done': self._to_tensor(self.not_done[idx])
        }
        
    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).to(self.device, non_blocking=True)
    
    def __len__(self) -> int:
        return self._size