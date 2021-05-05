import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import imageio
import envs

from torch.utils.data import DataLoader, ConcatDataset
from style_transfer.replay_memory_dataset import ReplayMemoryDataset
from style_transfer.ae import AE

MAX_LEN = 27

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env1-name', default="ant",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--env2-name', default="ant3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--model_path',
                    help='model path')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')

parser.add_argument('--hidden_dim', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--latent_dim', type=int, default=64,
                    help='Encoder latent dimension')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--actor_path', 
                    help='checkpoint training model every # steps')
parser.add_argument('--agent_memory1', default='data/ant.memory',
                    help='Path for saved replay memory')
parser.add_argument('--video_file_name', 
                    help='output file name')
args = parser.parse_args()


# Environment
# env = NormalizedActions(gym.make(args.env_name))
env1 = envs.load(args.env1_name)
env2 = envs.load(args.env2_name)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
device = torch.device("cuda" if args.cuda else "cpu")

dataset1 = ReplayMemoryDataset(args.agent_memory1)

s1 = dataset1[0][0].size(0)

style = torch.zeros(MAX_LEN)
style[:s1] = 1.

state_size = env1.observation_space.shape[0]

model = AE(state_size, state_size, args.hidden_dim, args.latent_dim).to(device=device)

model.load_model(args.model_path)

# Evaluation loop
total_numsteps = 0
avg_reward = 0.

render_env = env1
state = render_env.reset()

with imageio.get_writer(args.video_file_name, fps=30) as video:
    for idx, x, in enumerate(dataset1):
        state = x[0]
        x_hat = model(state, style)
        print(x_hat)
        #x_hat = torch.zeros_like(x_hat)
        render_env.set_to_observation(x_hat.detach().numpy())
        video.append_data(render_env.render('rgb_array'))

        if idx > 1000:
            break

env1.close()
env2.close()
