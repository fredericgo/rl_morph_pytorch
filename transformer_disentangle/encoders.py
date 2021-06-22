import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_num_limbs, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_num_limbs = max_num_limbs

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:, :x.size(1)]
        return self.dropout(x)


class StructureEncoding(nn.Module):
    def __init__(self, d_model, max_num_limbs, dropout=0.1, max_len=5000):
        super(StructureEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_num_limbs = max_num_limbs
        self.parent_embeddings = nn.Embedding(max_num_limbs, d_model)

    def forward(self, x):
        x = x + 1
        x = self.parent_embeddings(x)
        return self.dropout(x)



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
        print(ninp)
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
        self.mean_layer = nn.Linear(latent_size, ninp)
        self.var_layer = nn.Linear(latent_size, ninp)

    def forward(self, x):     
        x0 = x[:, :self.root_size].unsqueeze(1)
        z0 = self.root_projection(x0)
        x1 = x[:, self.root_size:]
        self.input_state = x1.reshape(self.batch_size, self.max_num_limbs, -1).permute(
            1, 0, 2
        )
        z1 = self.input_projection(self.input_state)
        z = torch.cat([z0, z1])
        z = z + self.pe(z) 
        z = self.encoder(z)
        mu = self.mean_layer(z)
        mu = mu.permute(1, 0, 2)
        var = self.var_layer(z)
        var = var.permute(1, 0, 2)
        return mu, var