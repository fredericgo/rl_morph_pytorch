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

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from transformer_vae.util import getGraphStructure
from transformer_vae.transformer_model import TransformerVAE
from transformer_vae import util
from transformer_vae.config import *
from transformer_vae.arguments import get_args
from transformer_vae.data_loader import PairedDataset

import envs

args = get_args()

device = torch.device("cuda" if args.cuda else "cpu")

env = gym.make(args.env1_name)
env.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

env_names = ["ant-v0", "ant3-v0", "ant_jump-v0", "ant3_jump-v0"]#"ant_a-v0", "ant_b-v0"]
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

xml_files = [env.xml for env in train_envs]
memory_files = [str(root_dir / f"data/{n[:-3]}.memory")  for n in env_names]
dataset = PairedDataset(
            memory_files,
            xml_files,
            dim_per_limb, 
            max_num_limbs,
            args.root_size)


model = TransformerVAE(
        dim_per_limb, 
        1, 
        args.msg_dim,
        args.batch_size,
        max_num_limbs,
        args).to(device=device)

#Tesnorboard
datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/{datetime_st}_VAE_{args.env1_name}_both'
writer = SummaryWriter(log_dir)

dataloader = DataLoader(dataset, 
                        batch_size=args.batch_size,
                        drop_last=True,
                        shuffle=True, num_workers=8)



def loss_function(data, x_hat):
    LossInfo = namedtuple('LossInfo',['rec_loss', 'KLD'])
    x_hat = x_hat.reshape([x_hat.shape[0], -1])
    x, _, _ = data
    x = x[:, ROOT_DIM:]
    reproduction_loss = torch.mean(torch.square((x_hat - x)))
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

        batch = [x.to(device) for x in batch]
        src, src2, top = batch
        batch = src, top, top
        x_hat = model(batch)
        loss, info = loss_function(batch, x_hat)
        overall_rec_loss += info.rec_loss.item()
        #overall_kl += info.KLD.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
    avg_loss = overall_rec_loss / (batch_idx)
    #avg_kl = overall_kl / (batch_idx)

    #writer.add_scalar('Model/logvar', torch.mean(logvar), epoch)

    writer.add_scalar('loss/rec', avg_loss, epoch)
    #writer.add_scalar('loss/KLD', avg_kl, epoch)


    print(f"\tEpoch {epoch + 1} completed!\t Average Loss: {avg_loss}")

    if epoch % args.checkpoint_interval == 0 and epoch > 0:
        model.save_model(log_dir)
        print("----------------------------------------")
        print(f"Save Model: {epoch} epoch.")
        print("----------------------------------------")