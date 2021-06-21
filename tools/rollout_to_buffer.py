import argparse
import datetime
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter

import gym
import envs
from rl.sac import SAC
from rl.replay_memory import ReplayMemory
from rl.model import GaussianPolicy, QNetwork, DeterministicPolicy

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ant_jump-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--num_steps', type=int, default=10000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--checkpoint_interval', type=int, default=500, 
                    help='checkpoint training model every # steps')

parser.add_argument('--actor_path', 
                    help='checkpoint training model every # steps')

parser.add_argument('--memory_path', 
                    help='checkpoint training model every # steps')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
device = torch.device("cuda" if args.cuda else "cpu")
policy = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], 
                        args.hidden_size, env.action_space).to(device)

policy.load_state_dict(torch.load(args.actor_path))


# Memory
memory = ReplayMemory(args.replay_size, args.seed)

def select_action(state):
    state = torch.FloatTensor(state).to(device).unsqueeze(0)
    _, _, action = policy.sample(state)
    return action.detach().cpu().numpy()[0]

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:

        action = select_action(state) # Sample action from policy

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

memory.save(args.memory_path)
env.close()

 