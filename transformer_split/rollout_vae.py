import argparse
import datetime
import gym
import envs

import numpy as np
import torch
import imageio

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from transformer_split.util import getGraphStructure
from transformer_split.replay_memory import ReplayMemoryDataset
from transformer_split.vae_model import VAE_Model

from torch.nn import functional as F
from transformer_vae import util

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env1-name', default="ant",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--env2-name', default="ant3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--model_path', default="runs/2021-05-19_13-46-41_VAE_ant-v0_both/",
                    help='model path')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--n_nodes', type=int, default=17,
                    help='max number of nodes in decoder')
parser.add_argument('--hidden_dim', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--latent_dim', type=int, default=128,
                    help='Encoder latent dimension')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--agent_memory1', default='data/ant_jump.memory',
                    help='Path for saved replay memory')
parser.add_argument('--video_file_name', default="ant_transfered.mp4",
                    help='output file name')
parser.add_argument('--msg_dim', type=int, default=32,
                        help='run on CUDA (default: False)')
parser.add_argument('--batch_size', type=int, default=1,
                        help='run on CUDA (default: False)')
parser.add_argument('--root_size', type=int, default=11,
                        help='root dimension')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='random seed (default: 123456)')
parser.add_argument(
    "--transformer_norm", default=0, type=int, help="Use layernorm",
)
parser.add_argument(
    "--attention_layers",
    default=3,
    type=int,
    help="How many attention layers to stack",
)
parser.add_argument(
    "--attention_heads",
    default=2,
    type=int,
    help="How many attention heads to stack",
)
parser.add_argument(
    "--attention_hidden_size",
    type=int,
    default=128,
    help="Hidden units in an attention block",
)

parser.add_argument(
    "--attention_embedding_size",
    type=int,
    default=128,
    help="Hidden units in an attention block",
)

parser.add_argument(
    "--dropout_rate",
    type=float,
    default=0.0,
    help="How much to drop if drop in transformers",
)

parser.add_argument(
        "--beta",
        type=float,
        default=1e-2,
        help="beta coefficient of KL divergence",
    )
parser.add_argument(
        "--discriminator_limiting_accuracy",
        type=float,
        default=0.6,
        help="beta coefficient of KL divergence",
    )
args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
device = torch.device("cuda" if args.cuda else "cpu")

env_names = ["ant-v0", "ant3-v0", "ant_a-v0", "ant_b-v0"]
train_envs = [gym.make(n) for n in env_names]
graphs = [getGraphStructure(e.xml) for e in train_envs]
# All environments have the same dimension per limb.
num_limbs = len(graphs[0])  #torso + body limbs
body_limbs = num_limbs - 1
dim_per_limb = int((train_envs[0].observation_space.shape[0] - args.root_size) / (body_limbs - 1))
max_num_limbs = max(len(g) for g in graphs)

args.dim_per_limb = dim_per_limb
args.max_num_limbs = max_num_limbs

root_dir = util.get_project_root()
render_env = train_envs[0]
render_graph = graphs[0]

memory_file = root_dir / f"data/ant_s1.memory" 
env = gym.make('ant_s1-v0')
dataset = ReplayMemoryDataset(
            memory_file, 
            getGraphStructure(env.xml),
            args)


vae_model = VAE_Model(args)

vae_model.load_model(args.model_path)

# Evaluation loop
total_numsteps = 0
avg_reward = 0.

state = render_env.reset()

dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
with imageio.get_writer(args.video_file_name, fps=30) as video:
    for idx, data, in enumerate(dataloader):
        
        x, src_top = data

        tgt_top = torch.full((max_num_limbs,), -1, dtype=torch.int32)
        tgt_top[:len(render_graph)] = torch.tensor(render_graph, dtype=torch.int32)
        tgt_top = tgt_top.unsqueeze(0)

        x_hat = vae_model.transfer(x, tgt_top)
        error = (x_hat - x).pow(2)
        render_env.set_to_observation(x_hat.detach().numpy().squeeze(0))
        video.append_data(render_env.render('rgb_array'))

        if idx > 1000:
            break


