import argparse
import os
import gym
import numpy as np
import pickle
import copy
import random

import torch
from torch.distributions import uniform

from sac import SAC
import ast

parser = argparse.ArgumentParser()
"""

model_noise - relative mass vs. noise probability
friction_noise - relative friction vs. noise probability
model_friction - relative mass vs. relative friction

"""
parser.add_argument('--eval_type', default='model',
                    choices=['model', 'model_noise', 'friction', 'friction_noise', "model_friction"])
parser.add_argument('--env', default="Swimmer-v2")
parser.add_argument("--optimizer")
parser.add_argument("--action_noise")

parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

base_dir = os.getcwd() + '/models/'

def eval_model(_env, alpha):
    total_reward = 0
    with torch.no_grad():
        state = torch.Tensor([_env.reset()])
        while True:
            action = agent.select_action(np.array(state), evaluate=True)
            if random.random() < alpha:
                action = noise.sample(action.shape).view(action.shape)

            state, reward, done, _ = _env.step(action)
            total_reward += reward

            #state = torch.Tensor([state])
            if done:
                break
    return total_reward


test_episodes = 100
for env_name in [args.env]:#os.listdir(base_dir):

    env = gym.make(env_name)
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    noise = uniform.Uniform(torch.Tensor([-1.0]), torch.Tensor([1.0]))

    basic_bm = copy.deepcopy(env.env.model.body_mass.copy())
    basic_friction = copy.deepcopy(env.env.model.geom_friction.copy())

    env_dir = base_dir + env_name
    for optimizer in [args.optimizer]: #['RMSprop', 'SGLD_thermal_0.01', 'SGLD_thermal_0.001', 'SGLD_thermal_0.0001', 'SGLD_thermal_1e-05']:
        for noise_type in [args.action_noise]:
            noise_dir = env_dir + '/'

            if os.path.exists(noise_dir):
                for subdir in sorted(os.listdir(noise_dir)):#[str(args.seed)]
                    results = {}

                    run_number = 0
                    dir = noise_dir + subdir #+ '/' + str(run_number)
                    print(dir)
                    if os.path.exists(noise_dir + subdir):#\
             # 		and not os.path.isfile(noise_dir + subdir + '/results_' + args.eval_type):
                        while os.path.exists(dir):
                            agent.load_model(basedir=dir)
                            if 'model_noise' in args.eval_type:
                                test_episodes = 10
                                for mass in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]: #np.linspace(0.8, 1.2, 10):
                                    if mass not in results:
                                        results[mass] = {}
                                    for alpha in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: #np.linspace(0, 0.5, 10):
                                        if alpha not in results[mass]:
                                            results[mass][alpha] = []
                                        for _ in range(test_episodes):
                                            for idx in range(len(basic_bm)):
                                                env.env.model.body_mass[idx] = basic_bm[idx] * mass
                                            r = eval_model(env, alpha)
                                            results[mass][alpha].append(r)

                            elif 'friction_noise' in args.eval_type:
                                test_episodes = 10
                                for friction in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]: #np.linspace(0.8, 1.2, 10):
                                    if friction not in results:
                                        results[friction] = {}
                                    for alpha in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: #np.linspace(0, 0.5, 10):
                                        if alpha not in results[friction]:
                                            results[friction][alpha] = []
                                        for _ in range(test_episodes):
                                            for idx in range(len(basic_friction)):
                                                env.env.model.geom_friction[idx] = basic_friction[idx] * friction
                                            r = eval_model(env, alpha)
                                            results[friction][alpha].append(r)

                            elif 'model_friction' in args.eval_type:
                                test_episodes = 10
                                for friction in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]: #np.linspace(0.8, 1.2, 10):
                                    if friction not in results:
                                        results[friction] = {}
                                    for mass in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]: #np.linspace(0, 0.5, 10):
                                        if mass not in results[friction]:
                                            results[friction][mass] = []
                                        for _ in range(test_episodes):
                                            for idx in range(len(basic_friction)):
                                                env.env.model.geom_friction[idx] = basic_friction[idx] * friction
                                            for idx in range(len(basic_bm)):
                                                env.env.model.body_mass[idx] = basic_bm[idx] * mass
                                            r = eval_model(env, 0)
                                            results[friction][mass].append(r)

                            elif 'model' in args.eval_type:
                                for mass in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]: #np.linspace(0.8, 1.2, 20):
                                    if mass not in results:
                                        results[mass] = []
                                    for _ in range(test_episodes):
                                        for idx in range(len(basic_bm)):
                                            env.env.model.body_mass[idx] = basic_bm[idx] * mass
                                        r = eval_model(env, 0)
                                        results[mass].append(r)
                            elif 'friction' in args.eval_type:
                                for friction in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]: #np.linspace(0.8, 1.2, 20):
                                    if friction not in results:
                                        results[friction] = []
                                    for _ in range(test_episodes):
                                        for idx in range(len(basic_friction)):
                                            env.env.model.geom_friction[idx] = basic_friction[idx] * friction
                                        r = eval_model(env, 0)
                                        results[friction].append(r)
                            else:
                                for alpha in np.linspace(0, 0.2, 20):
                                    if alpha not in results:
                                        results[alpha] = []
                                    for _ in range(test_episodes):
                                        r = eval_model(env, alpha)
                                        results[alpha].append(r)
                            run_number += 1
                            dir = noise_dir + subdir + '/' + str(run_number)
                        with open(noise_dir + subdir + '/results_' + args.eval_type, 'wb') as f:
                            pickle.dump(results, f)

env.close()
