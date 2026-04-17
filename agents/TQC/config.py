from dataclasses import dataclass

from shared.base import BaseConfig

@dataclass
class TQCConfig(BaseConfig):
    beta: float = 0.3
    gamma: float = 0.99
    critic_num: int = 5
    atom_num: int = 25
    critic_hidden: int = 512
    discard: int = 5
    device: str = 'cuda'
    
    # --- learning rate ---
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4