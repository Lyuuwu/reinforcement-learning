import importlib

from shared.base import BaseAgent

def build_agent(name: str, obs_dim, act_dim, max_act,
                config: str='default', overrides: dict | None = None, device='cuda') -> BaseAgent:
    module = importlib.import_module(f'agents.{name}')
    config_module = getattr(module, 'config', None)
    
    if config_module is None:
        raise AttributeError(f'agents "{name}" does not have file: config.py')
    
    config_fn = getattr(config_module, config, None)
    
    if config_fn is None:
        raise AttributeError(f'agents.{name}.config does not have function: "{config}"')
    
    cfg = config_fn()
    
    if overrides:
        cfg = cfg.override(**overrides)

    agent = module.build(
        obs_dim = obs_dim,
        act_dim = act_dim,
        max_act = max_act,
        config = cfg
    )
    
    return agent.to(device)