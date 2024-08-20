from multiprocessing import context
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
import envs.dmc as dmc
import envs.wrappers as wrappers
import agents
import plan
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel


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

def eval_episodes(env, num_episode, max_steps, num_envs, conf,
                  world_model: WorldModel, agent: agents.SoftActorCriticAgent, logger:Logger):
    world_model.eval()
    agent.eval()

    sum_reward = np.zeros(num_envs)
    current_obs = np.array([env.reset()['image']])

    context_obs = deque(maxlen=16)
    context_obs_cpu = deque(maxlen=16)
    context_action = deque(maxlen=16)
    context_reward = deque(maxlen=16)
    context_done = deque(maxlen=16)

    final_rewards = []
    # for total_steps in tqdm(range(max_steps//num_envs)):
    while True:
        # sample part >>>
        with torch.no_grad():
            if len(context_action) == 0:
                action = env.action_space.sample()
            else:
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=0)
                model_context_action = torch.Tensor([model_context_action]).cuda()
                if len(context_action) < 16:
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1)
                    )
                else:
                     action = plan.planning(
                        agent = agent,
                        world_model=world_model,
                        logger=None, 
                        plan_cfg=conf.Plan,
                        obs=context_obs_cpu, 
                        action=context_action, 
                        reward=context_reward, 
                        termination=context_done,
                        greedy=True
                        )

        context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
        context_obs_cpu.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W").cpu()/255)
        context_action.append(action)

        obs, reward, done, info = env.step(action)
        context_reward.append(reward)
        context_done.append(int(done))

        obs = np.array([obs['image']])
        reward = np.array([reward])
        done = np.array([int(done)])
        # cv2.imshow("current_obs", process_visualize(obs[0]))
        # cv2.waitKey(10)

        if done.any():
            for i in range(num_envs):
                final_rewards.append(sum_reward[i])
                # print(sum_reward[i])

                sum_reward[i] = 0
                current_obs = np.array([env.reset()['image']])
                context_obs.clear()
                context_obs_cpu.clear()
                context_action.clear()
                context_reward.clear()
                context_done.clear()

                if len(final_rewards) == num_episode:
                    # print("Mean reward: " + colorama.Fore.YELLOW + f"{np.mean(final_rewards)}" + colorama.Style.RESET_ALL)
                    return np.mean(final_rewards)

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        # <<< sample part


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-run_name", type=str, required=True)
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    seed = args.seed
    seed_np_torch(seed)

    # build env
    env = build_single_env(
        task=args.env_name, 
        action_repeat=conf.envs.action_repeat, 
        size=conf.envs.size,
        seed=args.seed
        )
    # initialize logger
    logger = Logger(path=f"plot/{version}/{args.env_name}/{args.seed}")

    # build and load model/agent
    import train
    action_dim = env.action_space.sample().shape[0]
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    root_path = f"ckpt/{version}/{args.run_name}/{seed}"

    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    steps = steps[:]
    results = []
    for step in tqdm(steps):
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        # eval
        episode_avg_return = eval_episodes(
            env=env, 
            num_episode=1,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            num_envs=1,
            conf=conf,
            world_model=world_model,
            agent=agent, 
            logger= logger
        )
        logger.log('eval_result', episode_avg_return.item())
        results.append([step, episode_avg_return])

    # create eval_result folder
    if not os.path.exists("curves"):
        os.makedirs("curves")

    if not os.path.exists(f"curves/{version}"):
        os.makedirs(f"curves/{version}")

    # rewrite results
    with open(f"curves/{version}/{args.run_name}.csv", "w") as fout:
        fout.write("step, episode_avg_return\n")
        for step, episode_avg_return in results:
            fout.write(f"{step},{episode_avg_return}\n")
