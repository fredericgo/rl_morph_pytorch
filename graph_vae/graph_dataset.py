from kinematics import mjcf_parser
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, memory_file, graph, max_nodes):
        self.raw_data = torch.load(memory_file)["state"]
        self.graph = graph
        self.num_node_features = 1
        self._max_nodes = max_nodes

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        data = self.raw_data[idx]
        g = self.skeleton.data_to_graph(data, self._max_nodes)
        return g  

if __name__ == "__main__":
    ds = GraphDataset("data/ant.memory", "envs/assets/ant.xml")
    for x in ds:
        print(x)