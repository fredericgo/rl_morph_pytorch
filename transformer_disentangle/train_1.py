import datetime
import gym
import numpy as np
import itertools
from itertools import cycle

import sys
sys.path.insert(0, '..')
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader

from transformer_disentangle.util import getGraphStructure
from transformer_disentangle.data_loader import PairedDataset
from transformer_disentangle import util
from transformer_disentangle.arguments import get_args
from transformer_disentangle.vae_model import VAE_Model

import envs

args = get_args()

device = torch.device("cuda" if args.cuda else "cpu")

env = gym.make(args.env1_name)
env.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

env_names = ["ant-v0", "ant3-v0", "ant_jump-v0", "ant3_jump-v0", "ant_a-v0", "ant_b-v0"]
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
            max_num_limbs
           )

vae_model = VAE_Model(args)

vae_model = vae_model.to(device)
#Tesnorboard
datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/{datetime_st}_VAE_{args.env1_name}_both'
#writer = SummaryWriter(log_dir)

dataloader = DataLoader(dataset, 
                        batch_size=args.batch_size,
                        drop_last=True,
                        shuffle=True, num_workers=3)
dataloader = cycle(dataloader)

epoch = 0

for epoch in range(args.epochs):
    overall_loss = 0
    kl_tot = 0
    rec_tot = 0
    loader = iter(dataloader)

    for iteration in range(int(len(dataset) / args.batch_size)):
        # A. run the auto-encoder reconstruction
        x1, x2, structure = next(loader)

        x1, x2, structure = x1.to(device), x2.to(device), structure.to(device)
        rec_loss, kl_loss = vae_model.train_recon(x1, x2, structure)    
        rec_tot += rec_loss
        kl_tot += kl_loss
        overall_loss += rec_loss + kl_loss
        
    avg_loss = overall_loss / iteration
    print(f"\tEpoch {epoch + 1} completed!\t Average Loss: {avg_loss}")
    writer.add_scalar('rec_loss', rec_tot / iteration, epoch)
    writer.add_scalar('kl_loss', kl_tot / iteration, epoch)

  
    if epoch % args.checkpoint_interval == 0 and epoch > 0:
        vae_model.save_model(log_dir)
        print("----------------------------------------")
        print(f"Save Model: {epoch} epoch.")
        print("----------------------------------------")