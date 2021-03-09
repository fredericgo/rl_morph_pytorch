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
from torch.utils.data import DataLoader
from rl.replay_memory import ReplayMemoryDataset
from graph_vae.skeleton_encoder import SkeletonEncoder
from graph_vae.motion_encoder import MotionEncoder
from graph_vae.motion_decoder import MotionDecoder
from graph_vae.model import VAE

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Ant-v3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--agent_memory1', default='data/ant.memory',
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

env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

dataset1 = ReplayMemoryDataset(args.agent_memory1)

state_size = env.observation_space.shape[0]
motion_encoder = MotionEncoder(state_size, 
                  hidden_dim=args.hidden_dim,
                  latent_dim=args.latent_dim)
skeleton_encoder = SkeletonEncoder(state_size, 
                  hidden_dim=args.hidden_dim,
                  latent_dim=args.latent_dim)
decoder = MotionDecoder(args.latent_dim,
                  hidden_dim=args.hidden_dim,
                  output_dim=state_size)
model = VAE(motion_encoder, skeleton_encoder, decoder)

#Tesnorboard
datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/{datetime_st}_VAE_{args.env_name}'
writer = SummaryWriter(log_dir)

dataloader = DataLoader(dataset1, batch_size=args.batch_size,
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

    for batch_idx, x, in enumerate(dataloader):
        state, action, reward, next_state, done = x

        optimizer.zero_grad()

        x_hat, mu, logvar = model(state)

        loss = loss_function(state, x_hat, mu, logvar)
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