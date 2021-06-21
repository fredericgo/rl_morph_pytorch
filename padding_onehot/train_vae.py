import argparse
import datetime
import gym
import numpy as np
import itertools

import sys
sys.path.insert(0, '..')
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from padding_onehot.replay_memory_dataset import ReplayMemoryDataset
from padding_onehot.skeleton_encoder import SkeletonEncoder
from padding_onehot.motion_encoder import MotionEncoder
from padding_onehot.motion_decoder import MotionDecoder
from padding_onehot.model import VAE
import envs

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env1-name', default="ant-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--env2-name', default="ant3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--agent_memory1', default='data/ant.memory',
                    help='Path for saved replay memory')
parser.add_argument('--agent_memory2', default='data/ant3.memory',
                    help='Path for saved replay memory')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='MLP hidden dimension')
parser.add_argument('--latent_dim', type=int, default=10,
                    help='Encoder latent dimension')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--checkpoint_interval', type=int, default=100, 
                    help='checkpoint training model every # steps')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")

env = gym.make(args.env1_name)
env.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

ant_paths = ["data/ant.memory", "data/ant_jump.memory"]
ant3_paths = ["data/ant3.memory", "data/ant3_jump.memory"]

datasets = []
for p in ant_paths:
    datasets.append(ReplayMemoryDataset(p, torch.tensor([1., 0.])))
for p in ant3_paths:
    datasets.append(ReplayMemoryDataset(p, torch.tensor([0., 1.])))
combined_dataset = ConcatDataset(datasets)

max_len = max(d[0][0].size(0) for d in datasets)

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
    out_x = out_x.to(device=device)
    out_y = out_y.to(device=device)
    return out_x, out_y

state_size = env.observation_space.shape[0]
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

#Tesnorboard
datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/{datetime_st}_VAE_{args.env1_name}'
writer = SummaryWriter(log_dir)

dataloader = DataLoader(combined_dataset, batch_size=args.batch_size,
                        collate_fn=collate_and_pad,
                        shuffle=True, num_workers=0)

def loss_function(x, x_hat, mean, log_var):
    LossInfo = namedtuple('LossInfo',['rec_loss', 'KLD'])

    #reproduction_loss = F.mse_loss(x_hat, x)
    reproduction_loss = torch.mean(torch.square((x_hat - x)))

    KLD = -0.5 * torch.mean(
        torch.sum(1+ log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)
    return reproduction_loss + KLD, LossInfo(reproduction_loss, KLD)

optimizer = Adam(model.parameters(), lr=args.lr)
print("Start training VAE...")
model.train()

epoch = 0

for epoch in range(args.epochs):
    overall_rec_loss = 0
    overall_kl = 0

    for batch_idx, batch, in enumerate(dataloader):
        optimizer.zero_grad()
    
        x_hat, mu, logvar = model(batch)
    
        loss, info = loss_function(batch[0], x_hat, mu, logvar)
        overall_rec_loss += info.rec_loss.item()
        overall_kl += info.KLD.item()
        
        loss.backward()
        optimizer.step()
    avg_loss = overall_rec_loss / (batch_idx)
    avg_kl = overall_kl / (batch_idx)
    writer.add_scalar('Model/logvar', torch.mean(logvar), epoch)

    writer.add_scalar('loss', avg_loss, epoch)

    print(f"\tEpoch {epoch + 1} completed!\t Average Loss: {avg_loss}")

    if epoch % args.checkpoint_interval == 0:
        model.save_model(log_dir)
        print("----------------------------------------")
        print(f"Save Model: {epoch} epoch.")
        print("----------------------------------------")