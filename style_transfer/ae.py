import os
import torch
import torch.nn as nn
from .motion_encoder import MotionEncoder
from .skeleton_encoder import SkeletonEncoder
from .motion_decoder import MotionDecoder

class F(nn.Module):
    def __init__(self, 
                 state_size, 
                 style_size, 
                 hidden_dim,
                 latent_dim):
        super(F, self).__init__()

        self.motion_encoder = MotionEncoder(state_size, hidden_dim, latent_dim)
        self.skeleton_encoder = SkeletonEncoder(style_size, hidden_dim, latent_dim)
        self.merger = nn.Linear(latent_dim*2, latent_dim)

    def forward(self, x, s):
        zs = self.skeleton_encoder(s)
        zm = self.motion_encoder(x)

        z = torch.cat((zs, zm), -1)
        return self.merger(z)


class AE(nn.Module):
    def __init__(self, 
                 state_size,
                 style_size, 
                 hidden_dim, 
                 latent_dim):

        super(AE, self).__init__()
        self.f = F(state_size, style_size, hidden_dim, latent_dim)
        self.decoder = MotionDecoder(latent_dim, hidden_dim, state_size)
                
    def forward(self, x, s):
        z = self.f(x, s)
        x_hat = self.decoder(z)
        return x_hat

    def save_model(self, path):
        model_path = os.path.join(path, 'ae_model')
        torch.save({
            "encoder": self.f.state_dict(),
            "decoder": self.decoder.state_dict()
        }, model_path)

    def load_model(self, path):
        model_path = os.path.join(path, 'ae_model')
        data = torch.load(model_path)
        self.f.load_state_dict(data['encoder'])
        self.decoder.load_state_dict(data['decoder'])

