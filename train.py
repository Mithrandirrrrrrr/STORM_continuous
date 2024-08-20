import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss
import envs.dmc as dmc
import envs.wrappers as wrappers


version = "0528"


def build_single_env(task, action_repeat, size, seed):
    env = dmc.DeepMindControl(
        task, action_repeat, size, seed=seed
        )
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, conf.envs.time_limit)
    # env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    # print('观测空间 = {}'.format(env.observation_space))
    # print('动作空间 = {}'.format(env.action_space))
    # print("Current env: " + colorama.Fore.YELLOW + f"{conf.envs.task}" + colorama.Style.RESET_ALL)
    return env


def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, demonstration_batch_size, batch_length, logger):
    obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
    world_model.update(obs, action, reward, termination, logger=logger)


@torch.no_grad()
def world_model_imagine_data(agent: agents.SoftActorCriticAgent, replay_buffer,
                            world_model: WorldModel,
                            imagine_batch_size, imagine_demonstration_batch_size,
                            imagine_context_length, imagine_batch_length,
                            single_start,
                            log_video, logger):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()

    if single_start:
        sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
            1, imagine_demonstration_batch_size, imagine_context_length)
        # print(sample_obs.shape, sample_action.shape, sample_reward.shape, sample_termination.shape)
        # torch.Size([1, 8, 3, 64, 64]) torch.Size([1, 8, 6]) torch.Size([1, 8]) torch.Size([1, 8])
        sample_obs = sample_obs.repeat(imagine_batch_size, 1, 1, 1, 1)
        sample_action = sample_action.repeat(imagine_batch_size, 1, 1)
        sample_reward = sample_reward.repeat(imagine_batch_size, 1)
        sample_termination = sample_termination.repeat(imagine_batch_size, 1)
        # torch.Size([num_samples, horizon, 3, 64, 64]) torch.Size([num_samples, horizon, 6]) torch.Size([num_samples, horizon]) torch.Size([num_samples, horizon])
    else:
        sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
            imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
        # print(sample_obs.shape, sample_action.shape, sample_reward.shape, sample_termination.shape)
        # torch.Size([num_samples, horizon, 3, 64, 64]) torch.Size([num_samples, horizon, 6]) torch.Size([num_samples, horizon]) torch.Size([num_samples, horizon])

    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size, # imagine_demonstration_batch_size = 0
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, reward_hat, termination_hat


def joint_train_world_model_agent(env, env_name, max_steps, num_envs,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.SoftActorCriticAgent,
                                  train_dynamics_every_steps, train_agent_every_steps,
                                  batch_size, demonstration_batch_size, batch_length,
                                  imagine_batch_size, imagine_batch_length,
                                  imagine_demonstration_batch_size,
                                  imagine_context_length,
                                  save_every_steps, single_start, logger):
    # create ckpt dir
    os.makedirs(f"ckpt/{version}/{args.n}/{args.seed}", exist_ok=True)

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs = np.array([env.reset()['image']])
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # sample and train
    for total_steps in tqdm(range(max_steps//num_envs)):
        # sample part >>>
        if replay_buffer.ready():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0 or total_steps % 200 == np.random.randint(0, 200):
                    action = env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=0)
                    model_context_action = torch.Tensor([model_context_action]).cuda()
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1)
                    )

            context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
            context_action.append(action)
        else:
            action = env.action_space.sample()

        obs, reward, done, info = env.step(action)

        obs = np.array([obs['image']])
        reward = np.array([reward])
        done = np.array([int(done)])
        # print("obs.shape:{}\n".format(obs.shape)) # (1, 64, 64, 3)
        # print("current_obs.shape:{}\n".format(current_obs.shape)) # (1, 64, 64, 3)
        # print("action.shape:{}\n".format(action.shape)) # (6,)
        # print("reward.shape:{}\n".format(reward.shape)) # (1,)
        # print("done.shape:{}\n".format(done.shape)) # (1,)        
        replay_buffer.append(current_obs, action, reward, done)

        if done.any():
            for i in range(num_envs):
                logger.log(f"sample/{env_name}_reward", sum_reward[i])
                # logger.log(f"sample/{env_name}_episode_steps", current_info["episode_frame_number"][i]//4)  # framskip=4
                # logger.log("replay_buffer/length", len(replay_buffer))
                sum_reward[i] = 0
                current_obs = np.array([env.reset()['image']])

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        # current_info = info
        # <<< sample part

        # train world model part >>>
        if replay_buffer.ready() and total_steps % (train_dynamics_every_steps//num_envs) == 0:
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                demonstration_batch_size=demonstration_batch_size,
                batch_length=batch_length,
                logger=logger
            )
        # <<< train world model part

        # train agent part >>>
        if replay_buffer.ready() and total_steps % (train_agent_every_steps//num_envs) == 0 and total_steps*num_envs >= 0:
            if total_steps % (save_every_steps//num_envs) == 0:
                log_video = True
            else:
                log_video = False

            imagine_latent, agent_action, imagine_reward, imagine_termination = world_model_imagine_data(
                agent=agent,
                replay_buffer=replay_buffer,
                world_model=world_model,
                imagine_batch_size=imagine_batch_size,
                imagine_demonstration_batch_size=imagine_demonstration_batch_size,
                imagine_context_length=imagine_context_length,
                imagine_batch_length=imagine_batch_length,
                single_start=single_start,
                log_video=log_video,
                logger=logger,
            )

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger
            )
        # <<< train agent part

        # save model per episode
        if total_steps % (save_every_steps//num_envs) == 0:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"ckpt/{version}/{args.n}/{args.seed}/world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), f"ckpt/{version}/{args.n}/{args.seed}/agent_{total_steps}.pth")


def build_world_model(conf, action_dim):
    return WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads
    ).cuda()


def build_agent(conf, action_dim):
    return agents.SoftActorCriticAgent(
        feat_dim=32*32+conf.Models.WorldModel.TransformerHiddenDim,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        tau=conf.Models.Agent.Tau,
        rho=conf.Models.Agent.Rho,
        epsilon=conf.Models.Agent.Epsilon
    ).cuda()


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-trajectory_path", type=str, required=True)
    parser.add_argument("-pretrain", type=int, required=True)
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=args.seed)
    # tensorboard writer
    logger = Logger(path=f"runs/{version}/{args.n}/{args.seed}")
    # copy config file
    shutil.copy(args.config_path, f"runs/{version}/{args.n}/{args.seed}/config.yaml")

    # distinguish between tasks, other debugging options are removed for simplicity
    if conf.Task == "JointTrainAgent":
        # build envs
        env = build_single_env(
            task=args.env_name, 
            action_repeat=conf.envs.action_repeat, 
            size=conf.envs.size,
            seed=args.seed
            )

        # get action_dim
        action_dim = env.action_space.sample().shape[0]

        # build world model and agent
        world_model = build_world_model(conf, action_dim)
        agent = build_agent(conf, action_dim)

        # load pretrain models
        if bool(args.pretrain):
            import glob
            pretrain_path = "pretrain_models"
            path = glob.glob(f"{pretrain_path}/world_model_*.pth")
            step = int(path[0].split("_")[-1].split(".")[0])
            world_model.load_state_dict(torch.load(f"{pretrain_path}/world_model_{step}.pth"))
            agent.load_state_dict(torch.load(f"{pretrain_path}/agent_{step}.pth"))
            print(f"Successfully loaded pretrain models of step {step} !")

        # build replay buffer
        replay_buffer = ReplayBuffer(
            action_dim = action_dim,
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU
        )

        # judge whether to load demonstration trajectory
        if conf.JointTrainAgent.UseDemonstration:
            print(colorama.Fore.MAGENTA + f"loading demonstration trajectory from {args.trajectory_path}" + colorama.Style.RESET_ALL)
            replay_buffer.load_trajectory(path=args.trajectory_path)

        # train
        joint_train_world_model_agent(
            env = env,
            env_name=args.env_name,
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            train_dynamics_every_steps=conf.JointTrainAgent.TrainDynamicsEverySteps,
            train_agent_every_steps=conf.JointTrainAgent.TrainAgentEverySteps,
            batch_size=conf.JointTrainAgent.BatchSize,
            demonstration_batch_size=conf.JointTrainAgent.DemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            batch_length=conf.JointTrainAgent.BatchLength,
            imagine_batch_size=conf.JointTrainAgent.ImagineBatchSize,
            imagine_batch_length=conf.JointTrainAgent.ImagineBatchLength, 
            imagine_demonstration_batch_size=conf.JointTrainAgent.ImagineDemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            imagine_context_length=conf.JointTrainAgent.ImagineContextLength,
            save_every_steps=conf.JointTrainAgent.SaveEverySteps,
            single_start=conf.Models.Agent.single_start,
            logger=logger
        )
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")
