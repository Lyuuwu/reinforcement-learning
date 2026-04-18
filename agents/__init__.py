import importlib

from shared.base import AgentBase

def build_agent(name: str, obs_dim, act_dim, max_act, config, device='cuda') -> AgentBase:
    module = importlib.import_module(f'agents.{name}')
    agent = module.build(
        obs_dim = obs_dim,
        act_dim = act_dim,
        max_act = max_act,
        config = config
    )
    return agent.to(device)