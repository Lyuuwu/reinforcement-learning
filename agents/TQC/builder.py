from shared.actors import GaussianActor
from .critic import Critic

from .agent import TQC
from .config import TQCConfig

def build(obs_dim: int, act_dim: int, max_act: float, config: TQCConfig):
    actor = GaussianActor(obs_dim, act_dim, max_act)
    critics = [Critic(obs_dim, act_dim, config.critic_hidden, config.atom_num) for _ in range(config.critic_num)]
    return TQC(
        actor=actor,
        critics=critics,
        config=config
    )
