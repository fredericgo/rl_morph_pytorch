import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GlobalAttention
from torch_scatter import scatter_mean
from skeleton_graph_vae.message_passing_layer import MessagePassingLayer


class GraphEncoder(torch.nn.Module):
    def __init__(self, num_inputs, hidden_dim, latent_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(num_inputs, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # mean and std projectors
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.var_layer = nn.Linear(hidden_dim, latent_dim)

        self.attention = GlobalAttention(nn.Linear(hidden_dim, 1))

    def forward(self, data):
        batch, x, edge_index, edge_weight = data.batch, data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index) #edge_weight=edge_weight)
        
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index) #edge_weight=edge_weight)
        #x = global_mean_pool(x, batch)
        x = self.attention(x, batch)
        mu = self.mean_layer(x)
        logvar = self.var_layer(x)
        var    = torch.exp(0.5 * logvar)  # takes exponential function
        epsilon = torch.rand_like(var)
        z = mu + var * epsilon   
        return mu

