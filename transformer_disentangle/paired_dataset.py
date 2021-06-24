import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from transformer_vae.util import getGraphStructure
from transformer_vae.config import *
import random


class PairedDataset(Dataset):
    def __init__(self, memory_file, topology_file, state_size, max_num_limbs):
        """
        state_size: dim per limb
        max_limbs: max number of limbs among all morphologies in the dataset
        """
        self.raw_data = torch.load(memory_file)["state"]
        self.topology = getGraphStructure(topology_file)
        self.state_size = state_size 
        self.max_num_limbs = max_num_limbs
        self.max_state_dim = ROOT_DIM + self.state_size * (self.max_num_limbs - 1)

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        data = self.raw_data[idx]
        output = torch.zeros(self.max_state_dim)
        output[:data.shape[0]] = torch.tensor(data)
        topology = torch.full((self.max_num_limbs,), -1, dtype=torch.int32)
        topology[:len(self.topology)] = torch.tensor(self.topology, dtype=torch.int32)
        data2 = random.choice(self.raw_data)
        output2 = torch.zeros(self.max_state_dim)
        output2[:data2.shape[0]] = torch.tensor(data2)
        return output, output2, topology
       

if __name__ == "__main__":
    ds = PairedDataset("data/ant.memory", "envs/xmls/ant.xml", 2, max_num_limbs=13)
    for x in ds:
        print(x)
        break