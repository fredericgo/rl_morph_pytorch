from kinematics import mjcf_parser
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from graph_vae import skeleton
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, memory_file, skeleton_file):
        raw_data = torch.load(memory_file)["state"]
        self.graph = skeleton.parse_skeleton(xml_file=skeleton_file)
        self._process_data(raw_data)

    def len(self):
        print(self.data["root"].shape[0])
        return self.data["root"].shape[0]

    def _process_data(self, raw_data):
        n_data = raw_data.shape[0]
        self.data = dict()
        for n in self.graph.nodes:
            start = self.graph.nodes.data()[n]['pos']
            end = start + self.graph.nodes.data()[n]['dof']

            if start < 0:
                data = np.zeros((n_data, 1), dtype=np.float32)
            else:
                data = raw_data[:, start:end]
            self.data[n] = data

    def __getitem__(self, idx):
        node_feature = torch.as_tensor([self.data[n][3] for n in self.graph.nodes], dtype=torch.float32)
        g = nx.to_directed(self.graph)
        g = nx.convert_node_labels_to_integers(g)
        edge_index = torch.as_tensor([[x[0], x[1]] for x in g.edges], dtype=torch.long)
        edge_feature = torch.as_tensor([g.edges[x[0], x[1]]["length"] for x in g.edges], dtype=torch.float32)
        return Data(x=node_feature, edge_index=edge_index, edge_attr=edge_feature)
        

if __name__ == "__main__":
    ds = GraphDataset("data/ant.memory", "envs/assets/ant.xml")
    for x in ds:
        print(x)