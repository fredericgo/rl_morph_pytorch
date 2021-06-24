import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformer_disentangle.embedding import PositionalEncoding, StructureEncoding

class StyleEncoder(nn.Module):
    def __init__(
        self,
        feature_size,
        latent_size, 
        ninp,
        nhead,
        nhid,
        nlayers,
        max_num_limbs,
        dropout=0.5,):
        super(StyleEncoder, self).__init__()

        self.model_type = "StyleEncoder"
        self.max_num_limbs = max_num_limbs

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(
            encoder_layers,
            nlayers,
            norm=nn.LayerNorm(ninp),
        )
        self.pe = PositionalEncoding(ninp, self.max_num_limbs)
        self.sturcture_emb = StructureEncoding(ninp, self.max_num_limbs)
        self.output_projection = nn.Linear(ninp, latent_size)

    def forward(self, topology):     
        # the first position is the root so we project it to latent space   
        z = self.pe(topology) + self.sturcture_emb(topology)
        z = self.encoder(z)
        z_root = z[:, 0]
        return self.output_projection(z_root)


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
        dropout=0.5,):
        super(PoseEncoder, self).__init__()

        self.model_type = "PoseEncoder"
        self.root_size = root_size
        self.batch_size = batch_size
        self.max_num_limbs = max_num_limbs
        self.root_projection = nn.Linear(root_size, ninp)
        self.input_projection = nn.Linear(feature_size, ninp)

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(
            encoder_layers,
            nlayers,
            norm=nn.LayerNorm(ninp),
        )
        self.pe = PositionalEncoding(ninp, self.max_num_limbs)
        self.mean_layer = nn.Linear(ninp, latent_size)
        self.var_layer = nn.Linear(ninp, latent_size)

    def forward(self, x):     
        x0 = x[:, :self.root_size].unsqueeze(1)
        z0 = self.root_projection(x0).permute(1, 0, 2)
        x1 = x[:, self.root_size:]
        self.input_state = x1.reshape(self.batch_size, self.max_num_limbs-1, -1).permute(
            1, 0, 2
        )
        z1 = self.input_projection(self.input_state)
        z = torch.cat([z0, z1])

        z = z + self.pe(z) 
        z = self.encoder(z)
        # take root only
        z = z[0]
        mu = self.mean_layer(z)
        var = self.var_layer(z)
        return mu, var

    def sample(self, x):
        mean, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        normal = Normal(mean, std)
        z = normal.rsample()
        return z
