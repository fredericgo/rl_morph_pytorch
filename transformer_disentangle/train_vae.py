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

from transformer_disentangle.util import getGraphStructure
from transformer_disentangle.replay_memory import ReplayMemoryDataset
from transformer_disentangle import util
from transformer_disentangle.encoders import StyleEncoder
from transformer_disentangle.config import *
from transformer_disentangle.arguments import get_args

import envs

args = get_args()

device = torch.device("cuda" if args.cuda else "cpu")

env = gym.make(args.env1_name)
env.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

env_names = ["ant-v0", "ant3-v0", "ant_jump-v0", "ant3_jump-v0"]
train_envs = [gym.make(n) for n in env_names]
graphs = [getGraphStructure(e.xml) for e in train_envs]
# All environments have the same dimension per limb.
num_limbs = len(graphs[0])  #torso + body limbs
body_limbs = num_limbs - 1
dim_per_limb = int((train_envs[0].observation_space.shape[0] - ROOT_DIM) / (body_limbs - 1))
max_num_limbs = max(len(g) for g in graphs)
args.max_num_limbs = max_num_limbs

root_dir = util.get_project_root()
datasets = []
for i, n in enumerate(env_names):
    memory_file = root_dir / f"data/{n[:-3]}.memory" 
    datasets.append(
        ReplayMemoryDataset(
            memory_file, 
            graphs[i],
            dim_per_limb, 
            max_num_limbs))

combined_dataset = ConcatDataset(datasets)


es = StyleEncoder(
        feature_size=dim_per_limb,
        latent_size=args.latent_dim,                  
        ninp=args.attention_embedding_size,
        nhead=args.attention_heads,
        nhid=args.attention_hidden_size,
        nlayers=args.attention_layers,
        max_num_limbs=args.max_num_limbs,
    )
#Tesnorboard
datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/{datetime_st}_VAE_{args.env1_name}_both'
writer = SummaryWriter(log_dir)

dataloader = DataLoader(combined_dataset, 
                        batch_size=args.batch_size,
                        drop_last=True,
                        shuffle=True, num_workers=8)

epoch = 0


for epoch in range(args.epochs):
    overall_rec_loss = 0

    for batch_idx, batch, in enumerate(dataloader):

        batch = [x.to(device) for x in batch]
        src, skeleton = batch
        zs = es(skeleton)

    avg_loss = overall_rec_loss / (batch_idx)
