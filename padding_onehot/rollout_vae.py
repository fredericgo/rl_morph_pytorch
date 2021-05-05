import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import imageio
import envs

from torch.utils.data import DataLoader, ConcatDataset
from padding_onehot.replay_memory_dataset import ReplayMemoryDataset
from padding_onehot.skeleton_encoder import SkeletonEncoder
from padding_onehot.motion_encoder import MotionEncoder
from padding_onehot.motion_decoder import MotionDecoder
from padding_onehot.model import VAE


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
parser.add_argument('--latent_dim', type=int, default=10,
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

dataset1 = ReplayMemoryDataset(args.agent_memory1, torch.tensor([1., 0.]))

state_size = env1.observation_space.shape[0]
motion_encoder = MotionEncoder(state_size, 
                  hidden_dim=args.hidden_dim,
                  latent_dim=args.latent_dim).to(device=device)
skeleton_encoder = SkeletonEncoder(2, 
                  hidden_dim=args.hidden_dim,
                  latent_dim=args.latent_dim).to(device=device)
decoder = MotionDecoder(args.latent_dim * 2,
                  hidden_dim=args.hidden_dim,
                  output_dim=state_size).to(device=device)
model = VAE(motion_encoder, skeleton_encoder, decoder)

model.load_model(args.model_path)

# Evaluation loop
total_numsteps = 0
avg_reward = 0.

render_env = env1
state = render_env.reset()

with imageio.get_writer(args.video_file_name, fps=30) as video:
    for idx, x, in enumerate(dataset1):
        state = x[0]
        label = x[5]
        x_hat, _, _ = model((state, label))
        render_env.set_to_observation(x_hat.detach().numpy())
        video.append_data(render_env.render('rgb_array'))

        if idx > 1000:
            break

env1.close()
env2.close()
