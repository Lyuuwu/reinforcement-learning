import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5 # SAC 原始是 -20
LOG_STD_MAX = 2

class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_action: float):
        super().__init__()
        
        self.act_dim = act_dim
        
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Linear(256, 256)
        )
        
        self.mean_head = nn.Linear(256, act_dim)
        self.log_std_head = nn.Linear(256, act_dim)
        self.max_action = max_action
        
    def forward(self, obs: torch.Tensor):
        feat = self.backbone(obs)
        
        mean = self.mean_head(feat)
    
        # ref1: https://docs.cleanrl.dev/rl-algorithms/sac/#implementation-details
        # ref2: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
        # 參考 CLEANRL 與 SAC 第三方 (star 數量約原始 SAC 的一半)
        log_std = self.log_std_head(feat)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        
        # 原始SAC:
        # log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
    
        std = log_std.exp()
        
        return mean, std
    
    def sample(self, obs):
        '''
        for train
        
        return action (B, D), log_prob (B, )
        '''
        mean, std = self.forward(obs)    # (B, D)
    
        dist = Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        
        log_prob = dist.log_prob(x).sum(dim=-1) # (B, )
        
        # Jacobian Correction
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1) # (B, D) -> (B, )
        log_prob -= action.shape[-1] * torch.log(self.max_action)
        
        return self.max_action * action, log_prob
    
    def act(self, obs):
        '''
        for eval
        
        return action
        '''
        mean, _ = self.forward(obs)
        return self.max_action * torch.tanh(mean)
