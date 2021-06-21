from kinematics import mjcf_parser
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, memory_file, skeleton, max_nodes):
        self.raw_data = torch.load(memory_file)["state"]
        self.skeleton = skeleton
        self.num_node_features = 1
        self._max_nodes = max_nodes

    def __len__(self):
        return self.raw_data.shape[0]

    def __getitem__(self, idx):
        data = self.raw_data[idx]
        g = self.skeleton.data_to_graph(data)
        return g
        print(g)
        node_feature = torch.as_tensor([self.data[n][3] for n in self.graph.nodes], dtype=torch.float32)
        g = nx.to_directed(self.graph)
        g = nx.convert_node_labels_to_integers(g)
        edge_index = torch.as_tensor([[x[0], x[1]] for x in g.edges], dtype=torch.long).t()
        edge_feature = torch.as_tensor([g.edges[x[0], x[1]]["length"] for x in g.edges], dtype=torch.float32)
        mask = torch.as_tensor([np.expand_dims(self.graph.nodes[x]["mask"], axis=-1) for x in self.graph.nodes])
        return Data(x=node_feature, edge_index=edge_index, edge_attr=edge_feature, mask=mask)
        

if __name__ == "__main__":
    ds = GraphDataset("data/ant.memory", "envs/assets/ant.xml")
    for x in ds:
        print(x)