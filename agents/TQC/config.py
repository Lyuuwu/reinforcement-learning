from dataclasses import dataclass

from shared.base import BaseConfig

@dataclass
class TQCConfig(BaseConfig):
    beta: float = 0.005
    gamma: float = 0.99
    critic_num: int = 5
    atom_num: int = 25
    critic_hidden: int = 512
    dropped: int = 0
    
    # --- learning rate ---
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
