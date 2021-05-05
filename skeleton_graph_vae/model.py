import os
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
                
    def forward(self, _input):
        zm = self.encoder(_input)

        x_hat = self.decoder(zm, _input)
        return x_hat#, mean, log_var

    def save_model(self, path):
        model_path = os.path.join(path, 'vae_model')
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict()
        }, model_path)

    def load_model(self, path):
        model_path = os.path.join(path, 'vae_model')
        data = torch.load(model_path)
        self.encoder.load_state_dict(data['encoder'])
        self.decoder.load_state_dict(data['decoder'])

