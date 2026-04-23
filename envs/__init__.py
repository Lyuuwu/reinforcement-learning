from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np

from .wrappers import(
    ActionRepeat, PixelObservation
)

# ==========================
# Config
# ==========================

@dataclass
class EnvConfig:
    seed: int | None = None
    
    # --- Episode Control ---
    max_episode_steps: int | None = None
    action_repeat: int = 1
    
    # --- Observation ---
    obs_type: Literal['state', 'pixels'] = 'state'
    image_size: tuple[int, int] = (64, 64)
    grayscale: bool = False
    frame_stack: int = 1
    
    # --- Action ---
    normalize_action: bool = True
    
    # --- Reward ---
    clip_reward: bool = False
    
    # --- domain kwargs ---
    domain_kwargs: dict[str, Any] = field(default_factory=dict)
    
# ==========================
# Parse Task
# ==========================

def parse_task(task: str) ->tuple[str, str]:
    '''
    mujoco:Halfcheetah -> ("mujoco", "HalfCheetah")
    
    dm_control:cheetah-run -> ("dm_control", "cheetah-run")
    '''
    
    if ':' not in task:
        raise ValueError(f'Format of Task should be "domain:task_name", but get "{task}"')
    
    domain, name = task.split(':', 1)
    
    return domain.strip(), name.strip()

# ==========================
# Base Builder
# ==========================

class BaseBuilder(ABC):
    def build(self, task_name: str, cfg: EnvConfig) -> gym.Env:
        env = self._make_base(task_name, cfg)
        env = self._apply_domain_wrappers(env, cfg)
        env = self._apply_universal_wrappers(env, cfg)
        self._seed(env, cfg.seed)
        return env
    
    @abstractmethod
    def _make_base(self, task_name: str, cfg: EnvConfig) -> gym.Env: ...
    
    def _apply_domain_wrappers(self, env: gym.Env, cfg: EnvConfig) -> gym.Env:
        return env
    
    def _apply_universal_wrappers(self, env: gym.Env, cfg: EnvConfig) -> gym.Env:
        if cfg.max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, cfg.max_episode_steps)
            
        if cfg.action_repeat > 1 and not getattr(env, '_action_repeat_handled', False):
            env = ActionRepeat(env, cfg.action_repeat)
            
        if cfg.normalize_action and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
        
        if cfg.clip_reward:
            env = gym.wrappers.TransformReward(env, np.sign)
        
        if cfg.frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=cfg.frame_stack)
        
        return env
    
    @staticmethod
    def _seed(env: gym.Env, seed: int | None) -> None:
        if seed is not None:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

# ==========================
# Domain Builder
# ==========================

class MuJoCoBuilder(BaseBuilder):
    def _make_base(self, task_name, cfg):
        return gym.make(f'{task_name}-v5', **cfg.domain_kwargs)
    
class DMControlBuilder(BaseBuilder):
    def _make_base(self, task_name, cfg):
        # e.g. walker_walk -> dm_control/walker-walk-v0
        domain, task = task_name.split('_', 1)
        kwargs = dict(cfg.domain_kwargs)
        if cfg.obs_type == 'pixels':
            kwargs['render_mode'] = 'rgb_array'
            kwargs['render_height'] = cfg.image_size[0]
            kwargs['render_width']  = cfg.image_size[1]
        return gym.make(f'dm_control/{domain}-{task}-v0', **kwargs)
    
    def _apply_domain_wrappers(self, env, cfg):
        if cfg.obs_type == 'state':
            env = gym.wrappers.FlattenObservation(env)
        else:
            h, w = cfg.image_size
            env = PixelObservation(env, h, w)
            if cfg.grayscale:
                env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
                
        return env
    
class AtariBuilder(BaseBuilder):
    def _make_base(self, task_name, cfg):
        return gym.make(f'ALE/{task_name}-v5', frameskip=1, **cfg.domain_kwargs)

    def _apply_domain_wrappers(self, env, cfg):
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=cfg.action_repeat,
            screen_size=cfg.image_size[0],
            terminal_on_life_loss=True,
            grayscale_obs=cfg.grayscale,
            scale_obs=False
        )
        env._action_repeat_handled = True
        return env
    
_BUILDERS: dict[str, BaseBuilder] = {
    "mujoco": MuJoCoBuilder(),
    "dm_control": DMControlBuilder(),
    "atari": AtariBuilder(),
}

# ==================================
# Public API
# ==================================

def make_env(task: str, cfg: EnvConfig | None = None) -> gym.Env:
    cfg = cfg or EnvConfig()
    domain, name = parse_task(task)
    if domain not in _BUILDERS:
        raise ValueError(f'unknown domain {domain!r}. avaliable: {list(_BUILDERS)}')
    return _BUILDERS[domain].build(name, cfg)

def make_env_fn(task: str, cfg: EnvConfig | None = None) -> Callable[[], gym.Env]:
    return partial(make_env, task, cfg)

def make_vec_env(
    task: str,
    num_envs: int,
    cfg: EnvConfig | None = None,
    async_mode: bool = True
) -> gym.vector.VectorEnv:
    cfg = cfg or EnvConfig()
    
    def thunk(worker_idx: int) -> gym.Env:
        worker_cfg = replace(
            cfg,
            seed = None if cfg.seed is None else cfg.seed + worker_idx
        )
        return make_env(task, worker_cfg)
    
    env_fns = [partial(thunk, i) for i in range(num_envs)]
    
    if async_mode and num_envs > 1:
        return gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    return gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)