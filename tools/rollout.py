import argparse
import datetime
import numpy as np
import itertools
import torch
import imageio
import envs
import gym

from rl.sac import SAC
from rl.replay_memory import ReplayMemory
from rl.model import GaussianPolicy, QNetwork, DeterministicPolicy

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ant_original-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_episodes', type=int, default=3, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--actor_path', 
                    help='checkpoint training model every # steps')
parser.add_argument('--video_file_name', 
                    help='output file name')
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


def select_action(state):
    state = torch.FloatTensor(state).to(device).unsqueeze(0)
    _, _, action = policy.sample(state)
    return action.detach().cpu().numpy()[0]

# Evaluation loop
total_numsteps = 0
avg_reward = 0.

with imageio.get_writer(args.video_file_name, fps=30) as video:
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False

        state = env.reset()
        video.append_data(env.render('rgb_array'))

        done = False
        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            video.append_data(env.render('rgb_array'))

            episode_reward += reward

            state = next_state
        avg_reward += episode_reward

        if i_episode > args.num_episodes:
            break

avg_reward /= args.num_episodes

print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}".format(args.num_episodes, round(avg_reward, 2)))
print("----------------------------------------")
    

env.close()

