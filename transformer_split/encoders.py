import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformer_split.embedding import PositionalEncoding, StructureEncoding

class PoseEncoder(nn.Module):
    def __init__(
        self,
        root_size,
        feature_size,
        latent_size, 
        batch_size, 
        ninp,
        nhead,
        nhid,
        nlayers,
        max_num_limbs,
        transformer_norm=True,
        dropout=0.5,):
        super(PoseEncoder, self).__init__()

        self.model_type = "PoseEncoder"
        self.ninp = ninp
        self.root_size = root_size
        self.batch_size = batch_size
        self.max_num_limbs = max_num_limbs
        self.root_projection = nn.Linear(root_size, ninp)
        self.input_projection = nn.Linear(feature_size, ninp)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(
            encoder_layers,
            nlayers,
            norm=nn.LayerNorm(ninp) if transformer_norm else None,
        )
        self.mean_layer = nn.Linear(ninp, latent_size)
        self.logvar_layer = nn.Linear(ninp, latent_size)
        self.content_layer = nn.Linear(ninp, latent_size)


    def forward(self, x):     
        input = x.reshape(self.batch_size, self.max_num_limbs-1, -1)
        root = torch.zeros(self.batch_size, 1, 2, device=x.device)
        input = torch.cat([root, input], dim=1)

        z = self.input_projection(input)
        z = z.permute(1, 0, 2)
        z = z * math.sqrt(self.ninp)
        z = self.encoder(z)
        # take root only
        mu = self.mean_layer(z)[0]
        logvar = self.logvar_layer(z)[0]
        c = self.content_layer(z)[0]
    
        std = torch.exp(0.5 * logvar)  # takes exponential function
        eps = std.new(std.size()).normal_()
        z = mu + std * eps
        return z, c, mu, logvar

    def sample(self, x):
        mean, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        normal = Normal(mean, std)
        z = normal.rsample()
        return z
