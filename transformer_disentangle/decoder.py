import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_disentangle.embedding import PositionalEncoding, StructureEncoding
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()

        self.model_type = "Decoder"
        self.root_size = root_size
        self.batch_size = batch_size
        self.max_num_limbs = max_num_limbs
    
        self.input_projection = nn.Linear(latent_size, ninp)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(
            decoder_layers,
            nlayers,
            norm=nn.LayerNorm(ninp),
        )
        self.pe = PositionalEncoding(ninp, self.max_num_limbs)
        self.root_projection = nn.Linear(ninp, root_size)
        self.output_projection = nn.Linear(ninp, feature_size)

      
    def forward(self, zs, zp):     
        tgt = self.pe.get_encoding(self.batch_size, self.max_num_limbs)
        tgt = tgt.permute(1, 0, 2)
        z = torch.cat([zs, zp], dim=1)
        z = self.input_projection(z)
        x = self.transformer_decoder(tgt, z)
        x0 = self.root_projection(x[:1]).reshape(self.batch_size, -1)
        x1 = self.output_projection(x[1:]).reshape(self.batch_size, -1)
        x = torch.cat([x0, x1], dim=1)
        return x


if __name__ == "__main__":
    de = Decoder(11, 2, 10, 128, 128, 2, 128, 3, 13)
    z = torch.ones(128, 10)
    x = de(z)
    print(x.shape)