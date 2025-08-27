import glob
import pickle as pkl
import lcm
import sys

from go2_gym_deploy.utils.deployment_runner import DeploymentRunner
from go2_gym_deploy.envs.lcm_agent import LCMAgent
from go2_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go2_gym_deploy.utils.command_profile import *

import pathlib

import torch.nn as nn

import numpy as np


class Actor(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dims=[512, 256, 128]):
        super(Actor, self).__init__()
        activation = nn.ELU()

        actor_layers = []
        actor_layers.append(nn.Linear(num_obs, hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

    def forward(self, x):
        return self.actor(x)

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], latent_dim=32, activation='elu'):
        super(MLPEncoder, self).__init__()
        activation = nn.ELU()   
        
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[l], latent_dim))
            else:
                encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)
    
    def forward(self, x):
        return self.encoder(x)


# lcm多播通信的标准格式
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[0]

# with open(logdir+"/parameters.pkl", 'rb') as file:
    with open(logdir+"/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

    print('Config successfully loaded!')

    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=1.0, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    from go2_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)
    print('Agent successfully created!')

    policy = load_policy(logdir)
    print('Policy successfully loaded!')

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    # print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

def load_policy(logdir):
    import os

    # try ------------------
    # body = torch.jit.load(logdir + '/checkpoints/body_latest.jit').to('cpu')
    global with_load_estimator
    
    if with_load_estimator:
        actor = Actor(num_obs=49+8, num_actions=12)
        actor.load_state_dict(torch.load(logdir + '/checkpoints/experiment/proposed_model/actor_0922_3.pth'))
        actor = actor.to('cpu')
        actor.eval()
        
    
    else:
        actor = Actor(num_obs=49, num_actions=12)
        actor.load_state_dict(torch.load('//home/unitree/program/legged_ball_catching/mujoco_test/model/rsl_rl_teacher_student/actor/actor_oracle38-5-15000.pth'))
        actor = actor.to('cpu')
        actor.eval()
        


    def policy(obs, info):
        global time_step
        time_step += 1
        action = actor(obs["obs"].to('cpu'))  # 直接喂 49 维

        if torch.max(action) > 5.5 or torch.min(action) < -5.5:
           print("======action========: ", action)        
        action = torch.clip(action, -6.7, 6.7)

        return action
    
    return policy


if __name__ == '__main__':
    # label = "gait-conditioned-agility/pretrain-v0/train"
    label = "gait-conditioned-agility/pretrain-go2/train"

    experiment_name = "example_experiment"
    # angular_velocity_list = []
    # body_quaternion_list = []
    obs_list = []
    time_step = 0

    with_load_estimator = False

    # default:
    # max_vel=3.5, max_yaw_vel=5.0
    load_and_run_policy(label, experiment_name=experiment_name, max_vel=0.7, max_yaw_vel=1.0)
