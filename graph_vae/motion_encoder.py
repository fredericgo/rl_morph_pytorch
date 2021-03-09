import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    def __init__(self, num_inputs, hidden_dim, latent_dim):
        super(MotionEncoder, self).__init__()

        # projection networks
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # mean and std projectors
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.var_layer = nn.Linear(hidden_dim, latent_dim)

        self.training = True

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mu = self.mean_layer(x)
        logvar = self.var_layer(x)
        var    = torch.exp(0.5 * logvar)  # takes exponential function
        epsilon = torch.rand_like(var)
        z = mu + var * epsilon   
        return z, mu, logvar