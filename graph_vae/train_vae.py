import argparse
import datetime
import gym
import numpy as np
import itertools

import sys
sys.path.insert(0, '..')
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from graph_vae.replay_memory_dataset import ReplayMemoryDataset
from graph_vae.skeleton_encoder import SkeletonEncoder
from graph_vae.motion_encoder import MotionEncoder
from graph_vae.motion_decoder import MotionDecoder
from graph_vae.model import VAE
import envs

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env1-name', default="ant",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--env2-name', default="ant3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--agent_memory1', default='data/ant.memory',
                    help='Path for saved replay memory')
parser.add_argument('--agent_memory2', default='data/ant3.memory',
                    help='Path for saved replay memory')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='MLP hidden dimension')
parser.add_argument('--latent_dim', type=int, default=64,
                    help='Encoder latent dimension')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--checkpoint_interval', type=int, default=100, 
                    help='checkpoint training model every # steps')
args = parser.parse_args()

env = envs.load(args.env1_name)
env.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

dataset1 = ReplayMemoryDataset(args.agent_memory1, torch.tensor([1., 0.]))
dataset2 = ReplayMemoryDataset(args.agent_memory2, torch.tensor([0., 1.]))
combined_dataset = ConcatDataset([dataset1, dataset2])

s1 = dataset1[0][0].size(0)
s2 = dataset2[0][0].size(0)
max_len = max(s1, s2)

def collate_and_pad(batch):
    B = len(batch)
    out_dims = (B, max_len)
    out_x = batch[0][0].new_full(out_dims, 0.)
    out_y = []
    for i, (state, _, _, _, _, skeleton) in enumerate(batch):
        length = state.size(0)
        out_x[i, :length, ...] = state
        out_y.append(skeleton)
    out_y = torch.stack(out_y)
    return out_x, out_y

state_size = env.observation_space.shape[0]
motion_encoder = MotionEncoder(state_size, 
                  hidden_dim=args.hidden_dim,
                  latent_dim=args.latent_dim)
skeleton_encoder = SkeletonEncoder(2, 
                  hidden_dim=args.hidden_dim,
                  latent_dim=args.latent_dim)
decoder = MotionDecoder(args.latent_dim * 2,
                  hidden_dim=args.hidden_dim,
                  output_dim=state_size)
model = VAE(motion_encoder, skeleton_encoder, decoder)

#Tesnorboard
datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/{datetime_st}_VAE_{args.env1_name}'
writer = SummaryWriter(log_dir)

dataloader = DataLoader(combined_dataset, batch_size=args.batch_size,
                        collate_fn=collate_and_pad,
                        shuffle=True, num_workers=0)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = F.mse_loss(x_hat, x)
    KLD = -0.5 * torch.mean(
        torch.sum(1+ log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)
    return reproduction_loss + KLD

optimizer = Adam(model.parameters(), lr=args.lr)
print("Start training VAE...")
model.train()

epoch = 0

for epoch in range(args.epochs):
    overall_loss = 0

    for batch_idx, batch, in enumerate(dataloader):
        optimizer.zero_grad()
    
        x_hat, mu, logvar = model(batch)
    
        loss = loss_function(batch[0], x_hat, mu, logvar)
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    avg_loss = overall_loss / (batch_idx * args.batch_size)
    writer.add_scalar('Model/logvar', torch.mean(logvar), epoch)

    writer.add_scalar('loss', avg_loss, epoch)

    print(f"\tEpoch {epoch + 1} completed!\t Average Loss: {avg_loss}")

    if epoch % args.checkpoint_interval == 0:
        model.save_model(log_dir)
        print("----------------------------------------")
        print(f"Save Model: {epoch} epoch.")
        print("----------------------------------------")