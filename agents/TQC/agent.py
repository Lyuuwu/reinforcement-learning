import collections
import itertools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.base import AgentBase
from .config import TQCConfig
from .critic import Critic
from shared.networks.actors import GaussianActor
from shared.networks.ema import EMA

class TQC(AgentBase):
    def __init__(
        self,
        actor: GaussianActor,
        critics: List[Critic],
        config: TQCConfig
    ):
        super().__init__()
        
        # --- models ---
        self.actor = actor
        self.critics = nn.ModuleList(critics)
        self.ema_critics = nn.ModuleList([EMA(critic, config.beta) for critic in critics])
        
        # --- alpha ---
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.alpha = self.log_alpha.exp()
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
        self.kN = (self.M - config.dropped) * self.N
        self.beta = config.beta
        self.gamma = config.gamma
        self.register_buffer(
            'tau',
            torch.tensor(
                [(2 * m - 1) / (2 * self.M) for m in range(1, self.M + 1)],
                dtype=torch.float32
            ).view(1, -1, 1)
        )

    def sample(self, obs):
        ''' return action (B, act_dim) , log_prob (B, ) '''
        return self.actor.sample(obs)

    @torch.no_grad()
    def act(self, obs):
        return self.actor.act(obs)
    
    def update(self, batch) -> None:
        # --- Policy Select Action ---
        actions, log_probs = self.sample(batch['obs'])
        
        # --- alpha ---
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self._stash('alpha/loss', alpha_loss)
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().detach()
        self._stash('alpha', self.alpha)

        # --- Actor Loss ---
        atoms = self._get_atoms(batch['obs'], actions, self.critics)
        actor_loss = self._actor_loss(log_probs, atoms)
        self._stash('actor/loss', actor_loss)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --- Critic Loss ---
        with torch.no_grad():
            next_actions, next_log_probs = self.sample(batch['next_obs'])
            
            ema_atoms = self._get_atoms(batch['next_obs'], next_actions, self.ema_critics)
            ema_atoms_flat, _ = ema_atoms.flatten(1).sort(dim=-1)   # (B, NM)
            ema_atoms_truncated = ema_atoms_flat[..., :self.kN]   # (B, kN)
            
            y = batch['reward'] + self.gamma * (ema_atoms_truncated - self.alpha * next_log_probs.unsqueeze(-1)) * batch['not_done']

        atoms = self._get_atoms(batch['obs'], batch['action'], self.critics)    # (B, NM)
        critic_loss = self._critic_loss(y, atoms).mean()
        self._stash('critic/loss', critic_loss)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- Target Network ---
        self._ema_update()

    def _actor_loss(self, log_probs: torch.Tensor, atoms: torch.Tensor):
        '''
        log_probs:  (B, )
        
        atoms:      (B, NM)
        '''
        
        return (self.alpha * log_probs - atoms.mean(dim=(-2, -1))).mean()

    def _critic_loss(self, y: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        '''
        y:      (B, kN)
        
        atoms:  (B, N, M)
        
        return (B, )
        '''
        
        #   y.unsqueeze(1).unsqueeze(1) -> (B, 1, 1, kN)
        #   atoms.unsqueeze(-1)         -> (B, N, M, 1)
        #   broadcast                   -> (B, N, M, kN)
        u = y.unsqueeze(1).unsqueeze(1) - atoms.unsqueeze(-1)
        
        loss = self._quantile_huber_loss(self.tau, u)
        total_loss = loss.sum(dim=(-3, -2, -1)) # (B, )
        
        return total_loss / (self.kN * self.M)

    def _get_atoms(self, obs, act, critics) -> torch.Tensor:
        tensors = [critic(obs, act) for critic in critics]  # List 長度為 N, 每個元素 (B, M)
        return torch.stack(tensors, dim=1)  # (B, N, M)
        
    def _quantile_huber_loss(self, tau: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        abs_u = u.abs()
        huber = torch.where(abs_u > 1.0, abs_u - 0.5, 0.5 * u * u)
        indicator = (u.detach() < 0).to(u.dtype)
        weight = (tau - indicator).abs()
        return weight * huber

    @torch.no_grad()
    def _ema_update(self) -> None:
        # lazy cache
        if not hasattr(self, '_ema_src'):
            src, tgt = [], []
            for critic, ema in zip(self.critics, self.ema_critics):
                src.extend(critic.parameters())
                tgt.extend(ema.ema_model.parameters())
            
            self._ema_src = src
            self._ema_tgt = tgt
        
        torch._foreach_lerp_(self._ema_tgt, self._ema_src, self.beta)