import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from transformer_structured.util import getGraphStructure

class ReplayMemoryDataset(Dataset):
    def __init__(self, memory_file, topology, args):
        """
        state_size: dim per limb
        max_limbs: max number of limbs among all morphologies in the dataset
        """
        self.raw_data = torch.load(memory_file)["state"]
        self.topology = topology
        self.state_size = args.dim_per_limb
        self.max_num_limbs = args.max_num_limbs
        self.max_state_dim = args.root_size + self.state_size * (self.max_num_limbs - 1)

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        data = self.raw_data[idx]
        output = torch.zeros(self.max_state_dim)
        output[:data.shape[0]] = torch.tensor(data)
        #num_limbs = torch.tensor(len(self.topology), dtype=torch.int32)
        topology = torch.full((self.max_num_limbs,), -1, dtype=torch.int32)
        topology[:len(self.topology)] = torch.tensor(self.topology, dtype=torch.int32)

        return output, topology
       

if __name__ == "__main__":
    ds = ReplayMemoryDataset("data/ant.memory", 10)
    for x in ds:
        print(x.shape)
        break