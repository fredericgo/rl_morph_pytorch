import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionDecoder(nn.Module):
    def __init__(self, num_inputs, hidden_dim, output_dim):
        super(MotionDecoder, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = F.relu(self.linear1(z))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        