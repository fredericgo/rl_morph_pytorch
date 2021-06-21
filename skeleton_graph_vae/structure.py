import numpy as np
import torch
from torch_geometric.data import Data
import skeleton_graph_vae.util as util
from gym.envs.mujoco import mujoco_env


class Structure:
    def __init__(self, xml_file):
        self.topology = util.getGraphStructure(xml_file)
        self.num_limbs = len(self.topology) 
        self._shift = 5 + 6 # root degrees of freedom
        self.state_dim = 2

    def data_to_graph(self, data):
        root = data[:self._shift]
        root = torch.tensor([root])
        body_data = data[self._shift:]
        node_feature = []
        node_feature.append(np.zeros(self.state_dim))
        print("body data:", body_data.shape)
        print("n_limbs", self.num_limbs)
        for i in range(self.num_limbs-1):
            node_feature.append(body_data[i * self.state_dim : (i+1) * self.state_dim])

        node_feature = torch.tensor(node_feature, dtype=torch.float32)

        edgelist = []
        for i in range(1, len(self.topology)):
            edgelist.append([i, self.topology[i]])
            edgelist.append([self.topology[i], i])
        edgelist = torch.LongTensor(edgelist).t()

        return Data(x=node_feature, edge_index=edgelist, root=root)
        