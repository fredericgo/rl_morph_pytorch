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
from style_transfer.replay_memory_dataset import ReplayMemoryDataset
from style_transfer.skeleton_template_dataset import SkeletonTemplateDataset
from style_transfer.skeleton_encoder import SkeletonEncoder
from style_transfer.motion_encoder import MotionEncoder
from style_transfer.motion_decoder import MotionDecoder
from style_transfer.ae import AE
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
parser.add_argument('--checkpoint_interval', type=int, default=10, 
                    help='checkpoint training model every # steps')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")

env = envs.load(args.env1_name)
env.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

dataset1 = ReplayMemoryDataset(args.agent_memory1)
dataset2 = ReplayMemoryDataset(args.agent_memory2)
combined_dataset = ConcatDataset([dataset1, dataset2])

s1 = dataset1[0][0].size(0)
s2 = dataset2[0][0].size(0)

skeleton_dataset = SkeletonTemplateDataset([s1, s2])

MAX_LEN = 27

def collate_and_pad(batch):
    B = len(batch)
    out_dims = (B, MAX_LEN)
    out_x = batch[0][0].new_full(out_dims, 0.)
    for i, (state, _, _, _, _) in enumerate(batch):
        length = state.size(0)
        out_x[i, :length, ...] = state
    out_x = out_x.to(device=device)
    return out_x

state_size = env.observation_space.shape[0]
model = AE(state_size, state_size, args.hidden_dim, args.latent_dim).to(device=device)



#Tesnorboard
datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/{datetime_st}_StyleAE'
writer = SummaryWriter(log_dir)

dataloader = DataLoader(combined_dataset, batch_size=args.batch_size,
                        collate_fn=collate_and_pad, drop_last=True,
                        shuffle=True, num_workers=2)
skeleton_loader = DataLoader(skeleton_dataset, batch_size=args.batch_size, num_workers=0)
skeleton_iter = iter(itertools.cycle(skeleton_loader))

def style_trasfer_loss(f, x, s, x_hat):
    dt = f(x_hat, s) - f(x, s)
    content_loss = torch.sum(torch.norm(dt, p=2, dim=-1))
    ds = f.skeleton_encoder(x_hat) - f.skeleton_encoder(s)
    style_loss = torch.sum(torch.norm(ds, p=2, dim=-1))
    return content_loss + style_loss

optimizer = Adam(model.parameters(), lr=args.lr)
print("Start training StyleAE...")
model.train()

epoch = 0

for epoch in range(args.epochs):
    overall_loss = 0

    for batch_idx, x, in enumerate(dataloader):
        s = next(skeleton_iter)
   
        optimizer.zero_grad()
        x_hat = model(x, s)
        
    
        loss = style_trasfer_loss(model.f,
                                  x, s, x_hat)
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    avg_loss = overall_loss / (batch_idx * args.batch_size)

    writer.add_scalar('loss', avg_loss, epoch)

    print(f"\tEpoch {epoch + 1} completed!\t Average Loss: {avg_loss}")

    if epoch % args.checkpoint_interval == 0:
        model.save_model(log_dir)
        print("----------------------------------------")
        print(f"Save Model: {epoch} epoch.")
        print("----------------------------------------")