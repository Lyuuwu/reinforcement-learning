import itertools
from typing import List

import torch
import torch.nn.functional as F

from shared.base import BaseAgent
from .config import TQCConfig
from .Critic import Critic
from shared.actors import GaussianActor
from shared.ema import EMA

class TQC(BaseAgent):
    def __init__(
        self,
        actor: GaussianActor,
        critics: List[Critic],
        config: TQCConfig
    ):
        super().__init__()
        
        # --- models ---
        self.actor = actor
        self.critics = critics
        self.ema_critics = [EMA(critic, config.beta) for critic in critics]
        
        # --- alpha ---
        self.log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
        self.alpha = self.log_alpha.exp().item()
        self.target_entropy = -self.actor.act_dim
        
        # --- optimizers ---
        self.actor_optimizer  = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            itertools.chain(*[c.parameters() for c in self.critics]),
            lr=config.critic_lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        
        # --- scales ---
        self.N = config.critic_num
        self.M = config.atom_num
        self.k = self.M - config.discard
        self.gamma = config.gamma
        self.tau = torch.tensor(
            [(2 * m - 1) / (2 * self.M) for m in range(1, self.M + 1)],
            dtype=torch.float32,
            device=config.device
        ).view(1, -1, 1)

    def sample(self, obs):
        ''' return action (B, act_dim) , log_prob (B, ) '''
        return self.actor.sample(obs)

    @torch.no_grad()
    def act(self, obs):
        return self.actor.act(obs)
    
    def update(self, batch) -> dict:
        metrics = {}
        
        # --- Critic Loss ---
        with torch.no_grad():
            next_actions, log_probs = self.sample(batch['next_obs'])
            
            ema_atoms = self._get_atoms(batch['next_obs'], next_actions, self.ema_critics)
            ema_atoms_flat, _ = ema_atoms.flatten(1).sort(dim=-1)   # (B, NM)
            ema_atoms_truncated = ema_atoms_flat[..., :self.k * self.N]   # (B, kN)
            
            y = batch['reward'] + self.gamma * (ema_atoms_truncated - self.alpha * log_probs.unsqueeze(-1)) * batch['not_done']

        atoms = self._get_atoms(batch['obs'], batch['action'], self.critics)    # (B, NM)
        critic_loss = self._critic_loss(y, atoms).mean()
        metrics['critics/loss'] = critic_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- Target Network ---
        for critic, ema_critic in zip(self.critics, self.ema_critics):
            ema_critic.update(critic)
        
        # --- Policy Loss ---
        actions, log_probs = self.sample(batch['obs'])
        atoms = self._get_atoms(batch['obs'], actions, self.critics)
        policy_loss = self._policy_loss(log_probs, atoms)
        metrics['policy/loss'] = policy_loss
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # --- alpha ---
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        metrics['alpha/loss'] = alpha_loss
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        return metrics

    def _critic_loss(self, y: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        '''
        y:      (B, kN)
        
        atoms:  (B, N, M)
        '''
        
        total_loss = 0
        for n in range(self.N):
            atoms_n = atoms[:, n, :]    # (B, M)
            # y.unsqueeze(1)        (B, 1, kN)
            # atoms_n.unsqueeze(2)  (B, M, 1)
            u = y.unsqueeze(1) - atoms_n.unsqueeze(2)
            loss = self._quantile_huber_loss(self.tau, u)
            total_loss += loss.sum(dim=(-2, -1))    # (B, )
        
        return total_loss / (self.k * self.N * self.M)    # (B, )
    
    def _policy_loss(self, log_probs: torch.Tensor, atoms: torch.Tensor):
        '''
        log_probs:  (B, )
        
        atoms:      (B, NM)
        '''
        
        return (self.alpha * log_probs - atoms.mean(dim=(-2, -1))).mean()

    def _get_atoms(self, obs, act, critics) -> torch.Tensor:
        tensors = [critic(obs, act) for critic in critics]  # List 長度為 N, 每個元素 (B, M)
        return torch.stack(tensors, dim=1)  # (B, N, M)
        
    def _quantile_huber_loss(self, tau: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        huber = F.huber_loss(u, torch.zeros_like(u), reduction='none', delta=1.0)
        return (tau - (u<0).float()).abs() * huber