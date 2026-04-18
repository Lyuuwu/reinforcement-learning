import argparse

import torch

from agents import build_agent

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Reinforcement Learning Projct')
    
    p.add_argument('--agent', type=str, required=True,
                   help='Agent name')
    
    p.add_argument('--task', type=str, required=True,
                   help='Task name (e.g. mujoco_ant, control_cheetah_run)')
    
    p.add_argument('--config', type=str, default='default',
                   help='experiment config')
    
    p.add_argument('--train_type', type=str, default='interleaved',
                   help='select training procedure')
    
    p.add_argument('--seed', type=int, default=0)
    
    p.add_argument('--device', type=str, default='auto',
                   help='"auto", "cpu", "cuda", "cuda:0"')
    
    p.add_argument('--save', type=str, default='auto',
                   help='Path to checkpoint .pt file')
    
    return p.parse_args()

def resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)

def get_trainer_class(trainer_type: str):
    raise NotImplementedError

def compose(agent_cfg, env_cfg) -> dict:
    raise NotImplementedError

if __name__ == '__main__':
    args = argparse()