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
from torch.utils.data import  ConcatDataset
from torch_geometric.data import DataLoader

from skeleton_graph_vae.structure import Structure
from skeleton_graph_vae.graph_dataset import GraphDataset

from skeleton_graph_vae.graph_encoder import GraphEncoder
from skeleton_graph_vae.graph_decoder import GraphDecoder
from skeleton_graph_vae.model import VAE
import envs

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env1-name', default="ant-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--env2-name', default="ant3-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--agent_memory1', default='data/ant.memory',
                    help='Path for saved replay memory')
parser.add_argument('--agent_memory2', default='data/ant3.memory',
                    help='Path for saved replay memory')
parser.add_argument('--hidden_dim', type=int, default=512,
                    help='MLP hidden dimension')
parser.add_argument('--latent_dim', type=int, default=10,
                    help='Encoder latent dimension')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--checkpoint_interval', type=int, default=100, 
                    help='checkpoint training model every # steps')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--n_nodes', type=int, default=13,
                    help='max number of nodes in decoder')
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")

env = gym.make(args.env1_name)
env.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

ant_paths = ["data/ant.memory"]

skeleton1 = Structure(env.xml)
datasets = []
for p in ant_paths:
    datasets.append(GraphDataset(p, skeleton1, max_nodes=args.n_nodes))

ant3_paths = ["data/ant3.memory"]
env2 = gym.make(args.env2_name)

skeleton_ant3 = Structure(env2.xml)

for p in ant3_paths:
    datasets.append(GraphDataset(p, skeleton_ant3, max_nodes=args.n_nodes))

combined_dataset = ConcatDataset(datasets)

input_size = 2

encoder = GraphEncoder(input_size, 
                        args.hidden_dim, 
                        args.latent_dim).to(device=device)

decoder = GraphDecoder(args.latent_dim,
                   args.hidden_dim,
                   input_size,
                   n_nodes=args.n_nodes).to(device=device)
model = VAE(encoder, decoder).to(device=device)

#Tesnorboard
datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/{datetime_st}_VAE_{args.env1_name}_both'
writer = SummaryWriter(log_dir)

dataloader = DataLoader(combined_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=8)

def loss_function(data, x_hat):
    LossInfo = namedtuple('LossInfo',['rec_loss', 'KLD'])
    reproduction_loss = torch.mean(torch.square((x_hat - data.x)))
    #KLD = -0.5 * torch.mean(
    #    torch.sum(1+ log_var - mean.pow(2) - log_var.exp(), dim=1), dim=0)
    return reproduction_loss, LossInfo(reproduction_loss, 0.)

optimizer = Adam(model.parameters(), lr=args.lr)
print("Start training VAE...")
model.train()

epoch = 0


for epoch in range(args.epochs):
    overall_rec_loss = 0
    #overall_kl = 0

    for batch_idx, batch, in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        x_hat = model(batch)
    
        loss, info = loss_function(batch, x_hat)
        overall_rec_loss += info.rec_loss.item()
        #overall_kl += info.KLD.item()

        loss.backward()
        optimizer.step()
    avg_loss = overall_rec_loss / (batch_idx)
    #avg_kl = overall_kl / (batch_idx)

    #writer.add_scalar('Model/logvar', torch.mean(logvar), epoch)

    writer.add_scalar('loss/rec', avg_loss, epoch)
    #writer.add_scalar('loss/KLD', avg_kl, epoch)


    print(f"\tEpoch {epoch + 1} completed!\t Average Loss: {avg_loss}")

    if epoch % args.checkpoint_interval == 0:
        model.save_model(log_dir)
        print("----------------------------------------")
        print(f"Save Model: {epoch} epoch.")
        print("----------------------------------------")