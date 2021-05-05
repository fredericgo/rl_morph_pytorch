import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU


class MessagePassingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MessagePassingLayer, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.m = torch.nn.Linear(2*out_channels, out_channels)

      
    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        return self.m(x)
