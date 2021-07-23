import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from transformer_disentangle.encoders import StyleEncoder, PoseEncoder
from transformer_disentangle.decoder import Decoder
from transformer_disentangle.discriminator import Discriminator


def kl_divergence(mu, logvar):
    return - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

def mse_loss(input, target):
    return (input - target).pow(2).mean()

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()

        style_enc = StyleEncoder(
            feature_size=args.dim_per_limb,
            latent_size=args.latent_dim,                  
            ninp=args.attention_embedding_size,
            nhead=args.attention_heads,
            nhid=args.attention_hidden_size,
            nlayers=args.attention_layers,
            max_num_limbs=args.max_num_limbs,
        )

        pose_enc = PoseEncoder(
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

        decoder = Decoder(
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

        discriminator = Discriminator(
            root_size=args.root_size,
            feature_size=args.dim_per_limb,
            max_num_limbs=args.max_num_limbs
       )

        self.add_module("style_enc", style_enc)
        self.add_module("pose_enc", pose_enc)
        self.add_module("decoder", decoder)
        self.add_module("discriminator", discriminator)

        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim

        encoder_parameters = list(self.style_enc.parameters()) + list(self.pose_enc.parameters())
        self.auto_encoder_optimizer = optim.Adam(
            encoder_parameters + list(self.decoder.parameters()),
            lr=args.lr,
        )

        self.discriminator_optimizer = optim.Adam(
            list(self.discriminator.parameters()),
            lr=args.lr,
        )

        self.generator_optimizer = optim.Adam(
            encoder_parameters + list(self.decoder.parameters()),
            lr=args.lr,
        )

        self.beta = args.beta
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.root_size = args.root_size


    def train_recon(self, x1, x2, structure):
        self.auto_encoder_optimizer.zero_grad()

        zs = self.style_enc(structure)
        zp_1, mean, logvar = self.pose_enc(x1)
        zp_2, mean, logvar = self.pose_enc(x2)

        x1_r = self.decoder(zp_1, zs)
        x2_r = self.decoder(zp_2, zs)
    
        kl_loss = kl_divergence(mean, logvar).mean()
        reconstruction_loss = mse_loss(x1_r, x1) + mse_loss(x2_r, x1)
        loss = reconstruction_loss + self.beta * kl_loss

        loss.backward()
        self.auto_encoder_optimizer.step()
        return reconstruction_loss, kl_loss


    def train_generator(self, x1, x3, structure3):
        self.generator_optimizer.zero_grad()

        zp_1, mean, logvar = self.pose_enc(x1)
        zs_3 = self.style_enc(structure3)

        xr_13 = self.decoder(zp_1, zs_3)

        kl_loss = kl_divergence(mean, logvar).mean()

        # True labels
        true_labels = torch.ones(self.batch_size, 1)
        true_labels = true_labels.to(self.device)

        d1 = self.discriminator(x3, xr_13)

        gen_loss_1 = F.binary_cross_entropy(d1, true_labels)

        z_random = torch.normal(0, 1, size=(self.batch_size, self.latent_dim))
        z_random = z_random.to(self.device)

        xr_r3 = self.decoder(z_random, zs_3)
        d2 = self.discriminator(x3, xr_r3)
        gen_loss_2 = F.binary_cross_entropy(d2, true_labels)

        generator_loss = gen_loss_1 + gen_loss_2 + self.beta * kl_loss

        generator_loss.backward()

        self.generator_optimizer.step()
        return gen_loss_1, gen_loss_2, kl_loss

    def train_discriminator(self, x1, x2, x3, structure3):
        self.discriminator_optimizer.zero_grad()

        true_labels = torch.ones(self.batch_size, 1)
        true_labels = true_labels.to(self.device)

        d_real = self.discriminator(x2, x3)

        disc_loss_real = F.binary_cross_entropy(d_real, true_labels)

        fake_labels = torch.zeros(self.batch_size, 1)
        fake_labels = fake_labels.to(self.device)

        zp_1, mean, logvar = self.pose_enc(x1)
        zs_3 = self.style_enc(structure3)
        xr_13 = self.decoder(zp_1, zs_3)
        d_fake = self.discriminator(x3, xr_13)

        disc_loss_fake = F.binary_cross_entropy(d_fake, fake_labels)

        discriminator_loss = disc_loss_real + disc_loss_fake
        discriminator_loss.backward()

        self.discriminator_optimizer.step()

        return discriminator_loss

    def save_model(self, path):
        model_path = os.path.join(path, 'vae_model')
        torch.save({
            "pose_encoder": self.pose_enc.state_dict(),
        }, model_path)
        torch.save({
            "style_encoder": self.style_enc.state_dict(),
        }, model_path)
        torch.save({
            "decoder": self.decoder.state_dict(),
        }, model_path)
        torch.save({
            "discriminator": self.discriminator.state_dict(),
        }, model_path)
       
    def load_model(self, path):
        model_path = os.path.join(path, 'vae_model')
        data = torch.load(model_path)
        self.encoder.load_state_dict(data['encoder'])