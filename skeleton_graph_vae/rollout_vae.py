import argparse
import datetime
import skeleton_graph_vae
import gym
import numpy as np
import torch
import imageio
import envs

#from torch.utils.data import DataLoader, ConcatDataset
from skeleton_graph_vae.graph_dataset import GraphDataset
from skeleton_graph_vae.graph_encoder import GraphEncoder
from skeleton_graph_vae.graph_decoder import GraphDecoder
from torch_geometric.data import DataLoader

from skeleton_graph_vae.replay_memory_dataset import ReplayMemoryDataset
from skeleton_graph_vae.model import VAE
from skeleton_graph_vae.skeleton import Skeleton
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env1-name', default="ant",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--env2-name', default="ant3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--model_path', default="runs/2021-04-20_23-29-10_VAE_ant/",
                    help='model path')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--n_nodes', type=int, default=17,
                    help='max number of nodes in decoder')
parser.add_argument('--hidden_dim', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--latent_dim', type=int, default=10,
                    help='Encoder latent dimension')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--agent_memory1', default='data/ant.memory',
                    help='Path for saved replay memory')
parser.add_argument('--video_file_name', default="graph_ant.mp4",
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

skeleton1 = Skeleton('envs/assets/ant.xml')
dataset1 = GraphDataset(args.agent_memory1, skeleton1, args.n_nodes)

input_size = 1
encoder = GraphEncoder(input_size, 
                  hidden_dim=args.hidden_dim,
                  latent_dim=args.latent_dim).to(device=device)

decoder = GraphDecoder(args.latent_dim,
                   args.hidden_dim,
                   input_size,
                   n_nodes=args.n_nodes).to(device=device)
model = VAE(encoder, decoder)

model.load_model(args.model_path)

def pad(x, dim=27):
    out = np.zeros(27)
    out[:x.shape[0]] = x
    return out

# Evaluation loop
total_numsteps = 0
avg_reward = 0.

render_env = env1
state = render_env.reset()

dataloader = DataLoader(dataset1, batch_size=1, num_workers=0)
with imageio.get_writer(args.video_file_name, fps=30) as video:
    for idx, x, in enumerate(dataloader):
        x_hat = model(x)
        x_out = skeleton1.attr_to_data(x, x_hat.detach())

        error = torch.square((x_hat - x.x))
        state = pad(x_out)
        render_env.set_to_observation(state)
        video.append_data(render_env.render('rgb_array'))

        if idx > 1000:
            break

env1.close()
env2.close()
