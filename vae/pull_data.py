import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from torch.utils.data import DataLoader
from rl.replay_memory import ReplayMemoryDataset


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--agent_memory1', default='data/ant.replay_buffer',
                    help='Path for saved replay memory')
args = parser.parse_args()


dataset1 = ReplayMemoryDataset(args.agent_memory1)

dataloader = DataLoader(dataset1, batch_size=4,
                        shuffle=True, num_workers=0)

for i, x in enumerate(dataloader):
    if i > 0: break
    print(x[0].size())
