import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_structured.embedding import PositionalEncoding, StructureEncoding
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
            norm=None,
        )
        pe = PositionalEncoding(ninp, self.max_num_limbs)
        self.add_module("pe", pe)
        self.structure_emb = StructureEncoding(ninp, self.max_num_limbs)
        self.root_projection = nn.Linear( ninp, root_size)
        self.output_projection = nn.Linear(ninp, feature_size)

      
    def forward(self, zp, structure):     
        structure = structure.transpose(1, 0)
        #tgt = self.pe(structure) 
        tgt = self.structure_emb(structure)
        z = self.input_projection(zp)
        x = self.transformer_decoder(tgt, z)
        x1 = self.output_projection(x[1:])
        x1 = x1.transpose(0, 1).reshape(self.batch_size, -1)
        return x1


if __name__ == "__main__":
    de = Decoder(11, 2, 10, 128, 128, 2, 128, 3, 13)
    z = torch.ones(128, 10)
    x = de(z)
    print(x.shape)