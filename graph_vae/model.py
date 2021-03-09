import os
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, motion_encoder, skeleton_encoder, decoder):
        super(VAE, self).__init__()
        self.motion_encoder = motion_encoder
        self.skeleton_encoder = skeleton_encoder
        self.decoder = decoder
                
    def forward(self, _input):
        x, y = _input
        zs, _, _ = self.skeleton_encoder(y)
        zm, mean, log_var = self.motion_encoder(x)

        x_hat = self.decoder(zs, zm)
        return x_hat, mean, log_var

    def save_model(self, path):
        model_path = os.path.join(path, 'vae_model')
        torch.save({
            "motion_encoder": self.motion_encoder.state_dict(),
            "skeleton_encoder": self.skeleton_encoder.state_dict(),
            "decoder": self.decoder.state_dict()
        }, model_path)

    def load_model(self, path):
        model_path = os.path.join(path, 'vae_model')
        data = torch.load(model_path)
        self.motion_encoder.load_state_dict(data['motion_encoder'])
        self.skeleton_encoder.load_state_dict(data['skeleton_encoder'])
        self.decoder.load_state_dict(data['decoder'])

