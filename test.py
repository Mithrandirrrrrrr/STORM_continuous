import torch
from torch import nn
from torch import distributions
from omegaconf import OmegaConf
import colorama
from tqdm import tqdm

import envs.dmc as dmc
import envs.wrappers as wrappers


def load_conf():
    conf = OmegaConf.load("config_files//continious.yaml")
    return conf


def build_env():
    env = dmc.DeepMindControl(
        conf.envs.task, conf.envs.action_repeat, conf.envs.size, seed=conf.envs.seed
        )
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, conf.envs.time_limit)
    # env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    print('观测空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print("Current env: " + colorama.Fore.YELLOW + f"{conf.envs.task}" + colorama.Style.RESET_ALL)
    return env


if __name__ == "__main__":
    conf = load_conf()
    env = build_env()
    current_obs = env.reset()
    # print("init_obs.shape:{}\n".format(current_obs['image'].shape))
    for i_steps in tqdm(range(5001)):
        current_action = env.action_space.sample()
        # print("current_action:{}\n".format(current_action))
        obs, reward, done, info = env.step(current_action)
        
        # print("obs['image'].shape:{}\n".format(obs['image'].shape))
        # print("obs['height']:{}\n".format(obs['height']))
        # print("obs['orientations'].shape:{}\n".format(obs['orientations'].shape))
        # print("obs['velocity'].shape:{}\n".format(obs['velocity'].shape))

        # print("reward:{}\n".format(reward))
        
        if done:
            print("i_steps:{}\n".format(i_steps))
            print("done:{}\n".format(done))
            env.reset()
            # break
    