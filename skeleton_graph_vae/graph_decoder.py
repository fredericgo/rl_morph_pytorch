import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from skeleton_graph_vae.message_passing_layer import MessagePassingLayer


class GraphDecoder(torch.nn.Module):
    def __init__(self, num_inputs, hidden_dim, output_dim, n_nodes):
        super(GraphDecoder, self).__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.num_inputs = num_inputs

        self.mlp_layer = nn.Sequential(
          nn.Linear(num_inputs, 256),
          nn.ReLU(),
          nn.Linear(256, n_nodes*num_inputs),
          nn.ReLU()
        )
        self.conv1 = GCNConv(num_inputs, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)


    def forward(self, z, data):
        edge_index = data.edge_index
        edge_weight = data.edge_attr
        z = self.mlp_layer(z).view(-1, self.num_inputs)
        x = self.conv1(z, edge_index)
        
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


        