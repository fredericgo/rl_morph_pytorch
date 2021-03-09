import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import imageio
import envs

from rl.sac import SAC
from rl.replay_memory import ReplayMemory
from rl.model import GaussianPolicy, QNetwork, DeterministicPolicy
from rl.replay_memory import ReplayMemoryDataset
from vae.encoder import Encoder
from vae.decoder import Decoder
from vae.model import VAE


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ant",
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
env = envs.load(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
device = torch.device("cuda" if args.cuda else "cpu")

dataset1 = ReplayMemoryDataset(args.agent_memory1)

state_size = env.observation_space.shape[0]
encoder = Encoder(state_size, 
                  hidden_dim=args.hidden_dim,
                  latent_dim=args.latent_dim)
decoder = Decoder(args.latent_dim,
                  hidden_dim=args.hidden_dim,
                  output_dim=state_size)
model = VAE(encoder, decoder)

model.load_model(args.model_path)

# Evaluation loop
total_numsteps = 0
avg_reward = 0.

state = env.reset()

with imageio.get_writer(args.video_file_name, fps=30) as video:
    for idx, x, in enumerate(dataset1):
        state, action, reward, next_state, done = x
        x_hat, _, _ = model(state)
        env.set_to_observation(x_hat.detach().numpy())
        video.append_data(env.render('rgb_array'))

        if idx > 1000:
            break

env.close()

