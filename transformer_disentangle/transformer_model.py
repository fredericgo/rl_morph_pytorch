import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from transformer_vae.config import *


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


def batched_index_select(input, dim, index):
	views = [input.shape[0]] + \
		[1 if i != dim else -1 for i in range(1, len(input.shape))]
	expanse = list(input.shape)
	expanse[0] = -1
	expanse[dim] = -1
	index = index.view(views).expand(expanse)
	return torch.gather(input, dim, index)


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
        

class TransformerModel(nn.Module):
    def __init__(
        self,
        feature_size,
        ninp,
        nhead,
        nhid,
        nlayers,
        max_num_limbs,
        dropout=0.5,
        transformer_norm=False,
    ):
        """This model is built upon https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            nlayers,
            norm=nn.LayerNorm(ninp) if transformer_norm else None,
        )
        self.encoder = nn.Linear(feature_size, ninp)
        self.ninp = ninp
        self.max_num_limbs = max_num_limbs
        self.pe = PositionalEncoding(ninp, self.max_num_limbs)
        self.sturcture_emb = StructureEncoding(ninp, self.max_num_limbs)

        self.transformer_decoder = TransformerDecoder(
            decoder_layers,
            nlayers,
            norm=nn.LayerNorm(ninp) if transformer_norm else None,
        )
        self.decoder = nn.Linear(ninp, feature_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_topology, tgt_topology):
        # add root postions and rotations (dummy)
        
        root = torch.zeros(1, src.size(1), 2)
        src = torch.cat([root, src], 0)
        encoded = self.encoder(src) * math.sqrt(self.ninp)
        #encoded += self.pe(src_topology) + self.sturcture_emb(src_topology)

        z = self.transformer_encoder(encoded)
        c = self.pe(tgt_topology) + self.sturcture_emb(tgt_topology)
        #if self.condition_decoder:
        #    output = torch.cat([output, src], axis=2)
        output = self.transformer_decoder(c, z)
        output = self.decoder(output)
        return output


class TransformerVAE(nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""

    def __init__(
        self,
        state_dim,
        action_dim,
        msg_dim,
        batch_size,
        max_num_limbs,
        args=None,
    ):
        super(TransformerVAE, self).__init__()
        self.num_limbs = 1
        self.max_num_limbs = max_num_limbs
        self.input_state = [None] * self.num_limbs
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.state_dim = state_dim

        self.encoder = TransformerModel(
            self.state_dim,
            args.attention_embedding_size,
            args.attention_heads,
            args.attention_hidden_size,
            args.attention_layers,
            self.max_num_limbs,
            args.dropout_rate,
            transformer_norm=args.transformer_norm,
        ).to(device)

    def forward(self, x, mode="train"):
        state, src_topology, tgt_topology = x

        self.clear_buffer()
        if mode == "inference":
            temp = self.batch_size
            self.batch_size = 1

        state = state[:, ROOT_DIM:]
        self.input_state = state.reshape(self.batch_size, self.max_num_limbs-1, -1).permute(
            1, 0, 2
        )
        src_topology = src_topology.transpose(1, 0)
        tgt_topology = tgt_topology.transpose(1, 0)

        self.z = self.encoder(
                    self.input_state, 
                    src_topology,
                    tgt_topology)
        # because of the permutation of the states, we need to unpermute the actions now so that the actions are (batch,actions)
        self.z = self.z.permute(1, 0, 2)
        
        # no torso
        self.z = self.z[:, 1:]
        if mode == "inference":
            self.batch_size = temp

        return torch.squeeze(self.z)

    def change_morphology(self, parents):
        self.parents = parents
        self.num_limbs = len(parents)
        self.input_state = [None] * self.num_limbs

    def clear_buffer(self):
        self.input_state = [None] * self.num_limbs

    def save_model(self, path):
        model_path = os.path.join(path, 'vae_model')
        torch.save({
            "encoder": self.encoder.state_dict(),
        }, model_path)
       
    def load_model(self, path):
        model_path = os.path.join(path, 'vae_model')
        data = torch.load(model_path)
        self.encoder.load_state_dict(data['encoder'])
