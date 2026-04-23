import numpy as np
import torch

from .train_base import TrainerBase
from .buffers import ReplayBuffer

class OffPolicyTrainer(TrainerBase):
    def __init__(self, *args, buffer: ReplayBuffer,
                 batch_size: int = 256,
                 updates_per_step: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = buffer
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step

    def _main_loop(self) -> None:
        obs = self._prefill()
        cfg = self.config

        while self.global_env_step < cfg.total_env_steps:
            # --- 收集一個 step ---
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                        device=self.device)
                action, _ = self.agent.sample(obs_t)
                action = action.cpu().numpy()

            next_obs, r, term, trun, _ = self.vec_env.step(action)

            for i in range(self.num_envs):
                self.buffer.push(obs[i], action[i], r[i], next_obs[i], bool(term[i]))

            obs = next_obs
            self.global_env_step += self.num_envs

            # --- 更新參數 ---
            if len(self.buffer) >= self.batch_size:
                for _ in range(self.updates_per_step):
                    self.agent.update(self.buffer.sample(self.batch_size))
                    self.global_update_step += 1

                if self.global_env_step % cfg.log_interval < self.num_envs:
                    metrics = self.agent.flush_metrics()
                    self.logger.log_print(metrics, step=self.global_env_step,
                                          prefix='train')

            # --- eval / save ---
            if self.global_env_step % cfg.eval_interval < self.num_envs:
                self.logger.log_print(self.evaluate(),
                                      step=self.global_env_step, prefix='eval')

            if self.global_env_step % cfg.save_interval < self.num_envs:
                self._save_checkpoint(tag='latest', include_buffer=False)
        
        # final eval
        self.logger.log_print(self.evaluate(),
                              step=self.global_env_step, prefix='eval')

    def _prefill(self):
        cfg = self.config
        need = max(cfg.warmup_steps - self.global_env_step, 0)
        obs, _ = self.vec_env.reset(seed=self.config.seed)
        
        if need == 0:
            print(f'[Prefill] warmup already satisfied (step={self.global_env_step})',
                  f'skipping ...')
            return obs
        
        print(f'[Prefill] collecting {need} random steps ...')
        

        collected = 0
        while collected < need:
            action = self.vec_env.action_space.sample()
            
            next_obs, r, term, trun, _ = self.vec_env.step(action)
            done = np.logical_or(term, trun)
        
            for i in range(self.num_envs):
                self.buffer.push(obs[i], action[i], r[i], next_obs[i], bool(term[i]))
            
            collected += self.num_envs
            obs = next_obs
            self.global_env_step += self.num_envs
                
        print(f'[Prefill] Done. Buffer has {len(self.buffer)} steps')
        
        return obs
    
    # --- buffer / optimizer 的 hook ---
    def _get_optim_state(self) -> dict:
        return {
            'actor': self.agent.actor_optimizer.state_dict(),
            'critic': self.agent.critic_optimizer.state_dict(),
            'alpha': self.agent.alpha_optimizer.state_dict(),
        }

    def _load_optim_state(self, state: dict) -> None:
        if not state: return
        self.agent.actor_optimizer.load_state_dict(state['actor'])
        self.agent.critic_optimizer.load_state_dict(state['critic'])
        self.agent.alpha_optimizer.load_state_dict(state['alpha'])

    def _get_buffer_state(self) -> dict:
        return {
            'obs': self.buffer.obs[:len(self.buffer)],
            'action': self.buffer.action[:len(self.buffer)],
            'reward': self.buffer.reward[:len(self.buffer)],
            'next_obs': self.buffer.next_obs[:len(self.buffer)],
            'not_done': self.buffer.not_done[:len(self.buffer)],
            'ptr': self.buffer._ptr,
            'size': self.buffer._size,
        }

    def _load_buffer_state(self, state: dict) -> None:
        n = state['size']
        self.buffer.obs[:n] = state['obs']
        self.buffer.action[:n] = state['action']
        self.buffer.reward[:n] = state['reward']
        self.buffer.next_obs[:n] = state['next_obs']
        self.buffer.not_done[:n] = state['not_done']
        self.buffer._ptr = state['ptr']
        self.buffer._size = state['size']