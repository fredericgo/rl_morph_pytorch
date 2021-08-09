import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformer_structured.embedding import PositionalEncoding, StructureEncoding

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
        pe = PositionalEncoding(ninp, self.max_num_limbs)
        self.add_module("pe", pe)
        self.structure_emb = StructureEncoding(ninp, self.max_num_limbs)
        self.output_projection = nn.Linear(ninp, latent_size)

    def forward(self, topology):     
        # the first position is the root so we project it to latent space   
        z = self.pe(topology) + self.structure_emb(topology)
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
            norm=None,
        )
        self.pe = PositionalEncoding(ninp, self.max_num_limbs)
        self.mean_layer = nn.Linear(ninp, latent_size)
        self.logvar_layer = nn.Linear(ninp, latent_size)

    def forward(self, x):     
        input = x.reshape(self.batch_size, self.max_num_limbs-1, -1)
        root = torch.zeros(self.batch_size, 1, 2, device=x.device)
        input = torch.cat([root, input], dim=1)
        
        z = self.input_projection(input)

        z = z.transpose(1, 0)
        z = z * math.sqrt(self.ninp)
        z = self.encoder(z)
        # take root only
        mu = self.mean_layer(z)[0]
        logvar = self.logvar_layer(z)[0]
    
        std = torch.exp(0.5 * logvar)  # takes exponential function
        eps = std.new(std.size()).normal_()
        z = mu + std * eps
        return z, mu, logvar

    def sample(self, x):
        mean, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        normal = Normal(mean, std)
        z = normal.rsample()
        return z
