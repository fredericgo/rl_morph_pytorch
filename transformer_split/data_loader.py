import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from transformer_split.util import getGraphStructure
import random


class PairedDataset(Dataset):
    def __init__(
        self, 
        memory_files:list, 
        topology_files:list, 
        state_size, 
        max_num_limbs,
        root_size):
        """
        state_size: dim per limb
        max_limbs: max number of limbs among all morphologies in the dataset
        """
        assert len(memory_files) == len(topology_files)
        self.state_size = state_size 
        self.max_num_limbs = max_num_limbs
        self.max_state_dim = root_size + self.state_size * (self.max_num_limbs - 1)
        data = self._load_data(memory_files, topology_files)
        self.data, self.labels, self.label_ranges, self.label_topology_dict = data

    def _load_data(self, memory_files, topology_files):
        data = []
        raw_data = [torch.load(f)['state'] for f in memory_files]
        file_to_label = {f: i for i, f in enumerate(topology_files)}

        label_topology_dict = {}
        for f in topology_files:
            top = getGraphStructure(f)
            #topology = np.zeros(self.max_num_limbs, dtype=np.int32)    
            topology = np.full((self.max_num_limbs,), -1, dtype=np.int32)

            topology[:len(top)] = top
            label_topology_dict[file_to_label[f]] = topology

        label_ranges = {}

        n = sum( x.shape[0] for x in raw_data )
        data = np.zeros((n, self.max_state_dim), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int32)

        curr_row = 0
        for i, x in enumerate(raw_data):
            rows = x.shape[0]
            cols = x.shape[1]
            data[curr_row:(curr_row+rows), :cols] = x
            labels[curr_row:(curr_row+rows)] = file_to_label[topology_files[i]]
            label_ranges[file_to_label[topology_files[i]]] = range(curr_row, (curr_row+rows))
            curr_row += rows

        return data, labels, label_ranges, label_topology_dict

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        topology = self.label_topology_dict[label]
        idx2 = random.choice(self.label_ranges[label])
        data2 = self.data[idx2]
        return data, data2, topology
       

if __name__ == "__main__":
    ds = PairedDataset(["data/ant.memory", "data/ant3.memory"], 
                       ["envs/xmls/ant.xml", "envs/xmls/ant3.xml"], 
                       2, 
                       max_num_limbs=13,
                       root_size=11)
    for d in ds:
        print(d)