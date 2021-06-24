import torch
import torch.nn as nn
from torch.nn import functional as F

from transformer_disentangle.encoders import StyleEncoder, PoseEncoder
from transformer_disentangle.decoder import Decoder

class VAE_Model(nn.Module):
    def __init__(self, args):
        self.style_enc = StyleEncoder(
            feature_size=args.dim_per_limb,
            latent_size=args.latent_dim,                  
            ninp=args.attention_embedding_size,
            nhead=args.attention_heads,
            nhid=args.attention_hidden_size,
            nlayers=args.attention_layers,
            max_num_limbs=args.max_num_limbs,
        )

        self.pose_enc = PoseEncoder(
            root_size=args.root_size,
            feature_size=args.dim_per_limb,
            latent_size=args.latent_dim,
            batch_size=args.batch_size,
            ninp=args.attention_embedding_size,
            nhead=args.attention_heads,
            nhid=args.attention_hidden_size,
            nlayers=args.attention_layers,
            max_num_limbs=args.max_num_limbs,
        )

        self.decoder = Decoder(
            root_size=args.root_size,
            feature_size=args.dim_per_limb,
            latent_size=args.latent_dim * 2,
            batch_size=args.batch_size,
            ninp=args.attention_embedding_size,
            nhead=args.attention_heads,
            nhid=args.attention_hidden_size,
            nlayers=args.attention_layers,
            max_num_limbs=args.max_num_limbs,
        )

        