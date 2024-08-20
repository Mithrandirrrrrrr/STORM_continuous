import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
import numpy as np
import math
from torch.cuda.amp import autocast
from functorch import combine_state_for_ensemble

from sub_models.functions_losses import SymLogTwoHotLoss
from utils import RunningScale


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, out_dim=1):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, out_dim)

        self.normlayer1 = nn.LayerNorm(hidden_size)
        self.normlayer2 = nn.LayerNorm(hidden_size)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, latent, action):
        x = torch.cat([latent, action], -1)
        x = self.linear1(x)
        x = F.silu(self.normlayer1(x))
        x = self.linear2(x)
        x = F.silu(self.normlayer2(x))
        x = self.linear3(x)
        return x
    

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.normlayer1 = nn.LayerNorm(hidden_size)
        self.normlayer2 = nn.LayerNorm(hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.silu(self.normlayer1(x))
        x = self.linear2(x)
        x = F.silu(self.normlayer2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std


class SoftActorCriticAgent(nn.Module):
    def __init__(self, feat_dim, hidden_dim, action_dim, gamma, tau, rho, epsilon) -> None:
        super().__init__()
        self.device = 'cuda'
        self.gamma = gamma
        self.tau = tau
        self.rho = rho
        self.epsilon = epsilon

        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32

        self.actor = PolicyNetwork(feat_dim, action_dim, hidden_dim).cuda()
        self.critic1 = SoftQNetwork(feat_dim, action_dim, hidden_dim).cuda()
        self.critic2 = SoftQNetwork(feat_dim, action_dim, hidden_dim).cuda()
        self.target_critic1 = copy.deepcopy(self.critic1).requires_grad_(False)
        self.target_critic2 = copy.deepcopy(self.critic2).requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, eps=1e-6)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4, eps=1e-6)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4, eps=1e-6)

        self.critic_scaler1 = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.critic_scaler2 = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.policy_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.scale = RunningScale(tau=0.01, use_amp=self.use_amp)

        # if training alpha
        self.target_entropy = torch.tensor(-action_dim, dtype=float, requires_grad=True, device='cuda')
        self.log_alpha = torch.tensor(np.log(1e-3), dtype=float, requires_grad=True, device='cuda')
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-5)

    @torch.no_grad()
    def update_target_critic(self):
        for target_param1, param1, target_param2, param2 in zip(self.target_critic1.parameters(), self.critic1.parameters(), self.target_critic2.parameters(), self.critic2.parameters()):
            target_param1.data.lerp_(param1.data, self.tau)
            target_param2.data.lerp_(param2.data, self.tau)

    def track_q_grad(self, mode=True):
        for p1, p2 in zip(self.critic1.parameters(), self.critic2.parameters()):
            p1.requires_grad_(mode)
            p2.requires_grad_(mode)

    def sample(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            mean, log_std = self.actor(latent) # torch.Size([batch_size, batch_length, 6])
            std = log_std.exp()
            
            z      = torch.randn_like(mean).cuda()
            action = torch.tanh(mean + std * z)
            log_prob = distributions.Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + self.epsilon) # torch.Size([batch_size, batch_length, 6])
            log_prob = log_prob.sum(dim=-1, keepdim=False) # torch.Size([batch_size, batch_length])
        return action, log_prob, z, mean, log_std

    @torch.no_grad()
    def sample_as_env_action(self, latent):
        self.eval()
        action, log_prob, z, mean, log_std = self.sample(latent)
        return action.detach().squeeze().cpu().numpy()# .squeeze(0).squeeze(1)
    
    @torch.no_grad()
    def _td_target(self, next_latent, reward, termination, gamma, alpha): # 计算TD目标
        next_actions, next_log_prob, epsilon, mean, log_std = self.sample(next_latent)
        target_q_value1 = self.target_critic1(next_latent, next_actions).squeeze(-1)
        target_q_value2 = self.target_critic2(next_latent, next_actions).squeeze(-1)
        next_value = torch.min(target_q_value1, target_q_value2) - alpha * next_log_prob
        td_target = reward + gamma * next_value * (1 - termination)
        return td_target

    def update(self, latent, action, reward, termination, logger=None):
        '''
        Update policy and value model
        '''
        # imagine_latent: torch.Size([batch_size, batch_length+1, 1536])
        # agent_action: torch.Size([batch_size, batch_length, 6])
        # imagine_reward: torch.Size([batch_size, batch_length])
        # imagine_termination: torch.Size([batch_size, batch_length])
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            batch_size, batch_length = action.shape[:2]
            predicted_q_value1 = self.critic1(latent[:, :-1], action).squeeze(-1)
            predicted_q_value2 = self.critic2(latent[:, :-1], action).squeeze(-1)

            # Training Q Function
            td_target = self._td_target(latent[:, 1:], reward, termination, self.gamma, self.log_alpha.exp())
            q_value_loss1, q_value_loss2 = 0, 0
            for t in range(batch_length):
                q_value_loss1 += self.soft_q_criterion1(predicted_q_value1[:, t], td_target[:, t]) * self.rho**t
                q_value_loss2 += self.soft_q_criterion2(predicted_q_value2[:, t], td_target[:, t]) * self.rho**t
            q_value_loss1 /= batch_length
            q_value_loss2 /= batch_length

        self.critic1_optimizer.zero_grad()
        self.critic_scaler1.scale(q_value_loss1).backward()
        self.critic_scaler1.unscale_(self.critic1_optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=100.0)
        self.critic_scaler1.step(self.critic1_optimizer)
        self.critic_scaler1.update()

        self.critic2_optimizer.zero_grad()
        self.critic_scaler2.scale(q_value_loss2).backward()
        self.critic_scaler2.unscale_(self.critic2_optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=100.0)
        self.critic_scaler2.step(self.critic2_optimizer)
        self.critic_scaler2.update()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # Training Policy Function
            self.track_q_grad(mode=False)
            new_action, log_prob, epsilon, mean, log_std = self.sample(latent[:, :-1])
            predicted_new_q_value = torch.min(self.critic1(latent[:, :-1], new_action), self.critic2(latent[:, :-1], new_action)).squeeze(-1)
            # scale
            self.scale.update(predicted_new_q_value[0])
            predicted_new_q_value = self.scale(predicted_new_q_value)
            
            policy_loss = 0
            for t in range(batch_length):
                policy_loss += (self.log_alpha.exp() * log_prob[:, t] - predicted_new_q_value[:, t]).mean() * self.rho ** t
            policy_loss /= batch_length
            
        self.actor_optimizer.zero_grad()
        self.policy_scaler.scale(policy_loss).backward()
        self.policy_scaler.unscale_(self.actor_optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100.0)
        self.policy_scaler.step(self.actor_optimizer)
        self.policy_scaler.update()
        self.track_q_grad(mode=True)

        # Update alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Training Target_Q Function
        self.update_target_critic()

        # entropy
        entropy = -log_prob.mean()

        if logger is not None:
            logger.log('SoftActorCritic/policy_loss', policy_loss.item())
            logger.log('SoftActorCritic/q_value_loss1', q_value_loss1.item())
            logger.log('SoftActorCritic/q_value_loss2', q_value_loss2.item())
            logger.log('SoftActorCritic/entropy', entropy.item())
            logger.log('SoftActorCritic/alpha', self.log_alpha.exp().item())
            # logger.log('ActorCritic/S', S.item())
            # logger.log('ActorCritic/norm_ratio', norm_ratio.item())
            # logger.log('ActorCritic/total_loss', loss.item())
