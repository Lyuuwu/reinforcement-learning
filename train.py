import argparse
import importlib
from pathlib import Path

import torch

from agents import build_agent
from envs import EnvConfig, make_vec_env, make_env
from shared.train_base import TrainerConfig
from shared.train_off_policy import OffPolicyTrainer
from shared.logger import JSONLLogger

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Reinforcement Learning Projct')
    
    p.add_argument('--agent', type=str, required=True,
                   help='Agent name')
    
    p.add_argument('--task', type=str, required=True,
                   help='Task name (e.g. mujoco_ant, control_cheetah_run)')
    
    p.add_argument('--config', type=str, default=None,
                   help='experiment config')
    
    p.add_argument('--train_type', type=str, default='off_policy',
                   help='select training procedure')
    
    p.add_argument('--seed', type=int, default=1000)
    
    p.add_argument('--num_envs', type=int, default=1)
    
    p.add_argument('--device', type=str, default='auto',
                   help='"auto", "cpu", "cuda", "cuda:0"')
    
    p.add_argument('--save_dir', default='runs/')
    
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint .pt file, "auto" or path')
    
    p.add_argument('--agent-override',   nargs='*', default=[], dest='agent_ov')
    p.add_argument('--env-override',     nargs='*', default=[], dest='env_ov')
    p.add_argument('--trainer-override', nargs='*', default=[], dest='trainer_ov')
    
    return p.parse_args()

def resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)

def build_configs(args):
    from experiments import load_experiment
    from shared.config_utils import apply_overrides, parse_cli_kv
    
    # --- dataclass default ---
    agent_mod   = importlib.import_module(f'agents.{args.agent}')
    agent_cfg   = agent_mod.Config()
    env_cfg     = EnvConfig(seed=args.seed)
    trainer_cfg = TrainerConfig(seed=args.seed, save_dir=args.save_dir)
    
    # --- experiments setting ---
    if args.config:
        exp = load_experiment(args.config)
        agent_cfg   = exp.get('agent', agent_cfg)
        env_cfg     = exp.get('env', env_cfg)
        trainer_cfg = exp.get('trainer', trainer_cfg)
    
    # --- CLI override ---
    agent_cfg   = apply_overrides(agent_cfg, parse_cli_kv(args.agent_ov))
    env_cfg     = apply_overrides(env_cfg, parse_cli_kv(args.env_ov))
    trainer_cfg = apply_overrides(trainer_cfg, parse_cli_kv(args.trainer_ov))
    
    return agent_cfg, env_cfg, trainer_cfg

def compose(args, agent_cfg, env_cfg, trainer_cfg, device) -> dict:
    # --- paths ---
    task_safe = args.task.replace(':', '_')
    run_dir = Path(trainer_cfg.save_dir) / f'[{args.agent}]' / f'[{args.agent}][{task_safe}]' / f'[{args.agent}][{task_safe}][{args.seed}]'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    trainer_cfg = trainer_cfg.override(save_dir=str(run_dir))
    
    # --- build env ---
    vec_env = make_vec_env(args.task, args.num_envs, env_cfg)
    eval_env = make_env(args.task, env_cfg)
    obs_dim = vec_env.single_observation_space.shape[0]
    act_dim = vec_env.single_action_space.shape[0]
    max_act = float(vec_env.single_action_space.high[0])
    
    # --- build agent ---
    agent = build_agent(args.agent, obs_dim, act_dim, max_act,
                        agent_cfg, device=device)
    
    # --- build buffer ---
    if args.train_type == 'off_policy':
        from shared.buffers import build_buffer
        buffer = build_buffer(
            kind=trainer_cfg.buffer_type,
            obs_dim=obs_dim, action_dim=act_dim,
            capacity=trainer_cfg.buffer_capacity,
            device=device,
            batch_size=trainer_cfg.batch_size
        )
    else:
        raise NotImplementedError
    
    # --- build logger ---
    logger = JSONLLogger(run_dir, args.agent, args.task, args.seed)
    logger.save_config({
        'agent_cfg': agent_cfg.to_dict(),
        'env_cfg': env_cfg.__dict__,
        'trainer_cfg': trainer_cfg.to_dict()
    })
    
    # --- build trainer ---
    if args.train_type == 'off_policy':
        trainer = OffPolicyTrainer(
            agent=agent, vec_env=vec_env, eval_env=eval_env,
            logger=logger, config=trainer_cfg, device=device,
            buffer=buffer,
            batch_size=trainer_cfg.batch_size,
            updates_per_step=trainer_cfg.updates_per_step
        )
    else:
        raise NotImplementedError
        
    return {
        'trainer': trainer,
        'agent': agent,
        'vec_env': vec_env,
        'eval_env': eval_env,
        'buffer': buffer,
        'logger': logger
    }

def main():
    # --- parse args ---
    args = parse_args()
    device = resolve_device(args.device)

    # --- build configs ---
    agent_cfg, env_cfg, trainer_cfg = build_configs(args)
    
    # --- compose ---
    components = compose(args, agent_cfg, env_cfg, trainer_cfg, device)
    
    param_count = sum(p.numel() for p in components['trainer'].agent.parameters())
    print('=' * 60)
    print(f'   Agent:   {args.agent}')
    print(f'   Task:    {args.task}')
    print(f'   Seed:    {args.seed}')
    print(f'   Device:  {device}')
    print(f'   Params:  {param_count}')
    print('=' * 60)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    try:
        components['trainer'].run()
    except KeyboardInterrupt:
        print('\n[Interrupted]')
        components['trainer']._save_checkpoint(tag='interrupted')
    finally:
        components['trainer']._log_vram_peak()
        components['vec_env'].close()
        components['eval_env'].close()

if __name__ == '__main__':
    main()