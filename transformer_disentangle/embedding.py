
import torch
import torch.nn as nn
import math

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

    def get_encoding(self, batch_size, length):
        return self.pe[:, :length].repeat(batch_size, 1, 1)

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