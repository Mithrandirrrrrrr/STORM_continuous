import torch
import numpy as np
import math

from sub_models.world_models import WorldModel
from agents import SoftActorCriticAgent


@torch.no_grad()
def select_topk(imagine_value, num_elites):
    index, value = torch.topk(imagine_value, num_elites, dim=0, largest=True, sorted=True)
    return index, value


@torch.no_grad()
def planning(agent:SoftActorCriticAgent, world_model:WorldModel, logger, plan_cfg, obs, action, reward, termination, greedy=True):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        world_model.eval()
        agent.eval()

        # Initialize starting state
        obs = np.stack(list(obs), axis=2)
        obs = torch.Tensor(obs).cuda()
        obs = obs.squeeze(0)
        action = np.stack(list(action), axis=0)
        action = torch.Tensor([action]).cuda()
        reward = np.stack(list(reward), axis=0)
        reward = torch.Tensor([reward]).cuda()
        termination = np.stack(list(termination), axis=0)
        termination = torch.Tensor([termination]).cuda()
        # torch.Size([1, 16, 3, 64, 64])
        pi_obs = obs.repeat(plan_cfg.num_pi_trajs, 1, 1, 1, 1)
        pi_action = action.repeat(plan_cfg.num_pi_trajs, 1, 1)
        gauss_obs = obs.repeat(plan_cfg.num_samples-plan_cfg.num_pi_trajs, 1, 1, 1, 1)
        gauss_action = action.repeat(plan_cfg.num_samples-plan_cfg.num_pi_trajs, 1, 1)
        start_obs = torch.cat([pi_obs, gauss_obs], dim=0) # torch.Size([num_samples, 16, 3, 64, 64])
        start_action = torch.cat([pi_action, gauss_action], dim=0)
        
        # Sample policy_action part
        pi_imagine_latent, pi_agent_action, pi_imagine_reward, pi_imagine_termination = world_model.imagine_data(
                agent, 
                pi_obs, 
                pi_action,
                imagine_batch_size=plan_cfg.num_pi_trajs,
                imagine_batch_length=plan_cfg.horizon,
                log_video=False,
                logger=logger
            ) # pi_agent_action: torch.size([num_pi_trajs, horizon, 6])
        
        # Initialize gaussian  paramaters
        action_dim = action.shape[-1]
        mean = torch.zeros(plan_cfg.horizon, action_dim, device='cuda')
        std = math.exp(plan_cfg.max_log_std) * torch.ones(plan_cfg.horizon, action_dim, device='cuda')

        # Initialize agent_actions
        agent_actions = torch.empty(plan_cfg.num_samples, plan_cfg.horizon, action_dim, device='cuda')
        agent_actions[:plan_cfg.num_pi_trajs] = pi_agent_action

        # Iterate MPPI
        for _ in range(plan_cfg.iterations):
            gauss_agent_action = torch.randn(plan_cfg.num_samples-plan_cfg.num_pi_trajs, plan_cfg.horizon, action_dim, device='cuda') \
                * std.unsqueeze(0) + mean.unsqueeze(0)
            gauss_agent_action = torch.tanh(gauss_agent_action)
            agent_actions[plan_cfg.num_pi_trajs:] = gauss_agent_action

            # Imagine data and calculate traj_value
            traj_value = world_model.estimate_value(
                agent, 
                start_obs, 
                start_action,
                agent_actions,
                agent.gamma, 
                imagine_batch_size=plan_cfg.num_samples,
                imagine_batch_length=plan_cfg.horizon
            ) # torch.size([num_samples])

            # Select topk
            topk_value, topk_idx = select_topk(traj_value, plan_cfg.num_elites) # torch.Size([num_elites])
            # print('topk_idx:',topk_idx)
            # print('topk_value:',topk_value)

            topk_action = agent_actions[topk_idx] # torch.Size([num_elites, horizon, 6])
            
            # Update parameters
            max_value = topk_value[0]
            score = torch.exp(plan_cfg.temperature*(topk_value - max_value)) # torch.Size([num_elites])
            score /= score.sum(0)
            mean = torch.sum(topk_action * score.unsqueeze(-1).unsqueeze(-1), dim=0) / (score.sum(0) + 1e-9) # torch.size([horizon, 6])
            std = torch.sqrt(torch.sum((topk_action - mean.unsqueeze(0)) ** 2 * score.unsqueeze(-1).unsqueeze(-1), dim=0) / (score.sum(0) + 1e-9)) \
                .clamp_(math.exp(plan_cfg.min_log_std), math.exp(plan_cfg.max_log_std)) # torch.size([horizon, 6])
        # assert 0==1
        # Select action
        '''
        score = score.cpu().numpy()
        actions = topk_action[np.random.choice(np.arange(score.shape[0]), p=score)] # torch.Size([horizon, 6])
        #print("topk_value, topk_idx", topk_value, topk_idx)
        #print("topk_action", topk_action)
        #print("score", score)
        #print("actions", actions)
        
        a, std = actions[0].float(), std[0].float()
        if greedy:
            pass
        else:
            a += std * torch.randn(action_dim, device='cuda')
        return a.cpu().numpy()
        '''
        chosen_action = topk_action[0]
        first_action = chosen_action[0]
        if greedy:
            pass
        else:
            first_action += std[0] * torch.randn(action_dim, device='cuda')
            first_action = first_action.clamp_(-1, 1)
        return first_action.float().detach().cpu().numpy()
        