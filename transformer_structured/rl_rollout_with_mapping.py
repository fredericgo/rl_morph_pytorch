import argparse
import datetime
import gym
import envs

import numpy as np
import torch
import imageio
import itertools

from rl.model import GaussianPolicy, QNetwork, DeterministicPolicy

from transformer_structured.util import getGraphStructure
from transformer_structured.vae_model import VAE_Model
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
parser.add_argument('--policy_hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--latent_dim', type=int, default=128,
                    help='Encoder latent dimension')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--agent_memory1', default='data/ant_jump.memory',
                    help='Path for saved replay memory')
parser.add_argument('--video_file_name', default="ant_turn.mp4",
                    help='output file name')
parser.add_argument('--msg_dim', type=int, default=32,
                        help='run on CUDA (default: False)')
parser.add_argument('--batch_size', type=int, default=1,
                        help='run on CUDA (default: False)')
parser.add_argument('--actor_path', 
                    help='checkpoint training model every # steps')
parser.add_argument('--num_episodes', type=int, default=3, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--root_size', type=int, default=11,
                        help='root dimension')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                        help='random seed (default: 123456)')
parser.add_argument(
    "--transformer_norm", default=0, type=int, help="Use layernorm",
)
parser.add_argument(
    "--beta",
    type=float,
    default=.1,
    help="beta coefficient of KL divergence",
)

parser.add_argument(
    "--gradient_penalty",
    type=float,
    default=10,
    help="beta coefficient of KL divergence",
)

parser.add_argument(
    "--discriminator_limiting_accuracy",
    type=float,
    default=0.7,
    help="beta coefficient of KL divergence",
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
args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
device = torch.device("cuda" if args.cuda else "cpu")

env_names = ["ant-v0", "ant3-v0", "ant_a-v0"]
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
render_env = train_envs[2]
render_topology = graphs[2]
render_limbs = len(render_topology)

expert_env = train_envs[0]
expert_topology = graphs[0]

policy = GaussianPolicy(
            expert_env.observation_space.shape[0], 
            expert_env.action_space.shape[0], 
            args.policy_hidden_size, 
            expert_env.action_space).to(device)

policy.load_state_dict(torch.load(args.actor_path))


vae_model = VAE_Model(args)

vae_model.load_model(args.model_path)

def pad_state(data, state_size, max_num_limbs):
    max_dim = args.root_size + state_size * (max_num_limbs - 1)
    output = torch.zeros(max_dim)
    output[:data.shape[0]] = torch.tensor(data)
    return output

def pad_topology(top, max_num_limbs):
    topology = torch.full((max_num_limbs,), -1, dtype=torch.int32)
    topology[:len(top)] = torch.tensor(top, dtype=torch.int32)
    return topology

# Evaluation loop
total_numsteps = 0
avg_reward = 0.
state = render_env.reset()

with imageio.get_writer(args.video_file_name, fps=30) as video:

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False

        state = render_env.reset()
        video.append_data(render_env.render('rgb_array'))

        done = False
        while not done:
            state = pad_state(state, dim_per_limb, max_num_limbs).unsqueeze(0)
            src_topology = pad_topology(render_topology, max_num_limbs).unsqueeze(0)
            tgt_topology = pad_topology(expert_topology, max_num_limbs).unsqueeze(0)

            x_hat = vae_model.transfer(state, tgt_topology)
            x_hat = x_hat.detach().cpu()
            x_hat = x_hat[:(render_limbs-1)]
            x_hat = torch.FloatTensor(x_hat).to(device).unsqueeze(0)
            action, _, _ = policy.sample(x_hat)
            action = action.detach().cpu().numpy()[0]

            next_state, reward, done, _ = render_env.step(action[0][:7])
            video.append_data(render_env.render('rgb_array'))

            episode_reward += reward

            state = next_state
        avg_reward += episode_reward

        if i_episode > args.num_episodes:
            break


