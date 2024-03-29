import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np

from transformer_structured.encoders import StyleEncoder, PoseEncoder
from transformer_structured.decoder import Decoder
from transformer_structured.discriminator import Discriminator


def kl_divergence(mu, logvar):
    return - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

def mse_loss(input, target):
    return (input - target).pow(2).mean()

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L    


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()

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
            dropout=args.dropout_rate
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
            dropout=args.dropout_rate
        )

        discriminator = Discriminator(
            root_size=args.root_size,
            feature_size=args.dim_per_limb,
            max_num_limbs=args.max_num_limbs
       )

        self.add_module("pose_enc", pose_enc)
        self.add_module("decoder", decoder)
        self.add_module("discriminator", discriminator)

        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim

        encoder_parameters = list(self.pose_enc.parameters())
        self.auto_encoder_optimizer = optim.Adam(
            encoder_parameters + list(self.decoder.parameters()),
            lr=args.ae_lr,
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
        self.discriminator_limiting_accuracy = args.discriminator_limiting_accuracy
        self.gp_weight = args.gradient_penalty

        self.beta_schedule = frange_cycle_linear(0, args.beta, args.epochs, 3, 1)


    def _gradient_penalty(self, D, real_data, generated_data):
        real_data = torch.cat(real_data, dim=-1)
        generated_data = torch.cat(generated_data, dim=-1)
        batch_size = real_data.size(0)
        d = int(real_data.size(1) / 2)
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, device=real_data.device, requires_grad=True)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(generated_data.device)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = torch.split(interpolated, [d, d], dim=-1)
    
        # Calculate probability of interpolated examples
        prob_interpolated = D(*interpolated)
        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size(), 
                                                       device=real_data.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()
    def split_root_body(self, x):
        x_root = x[:, :self.root_size]
        x_body = x[:, self.root_size:]
        return x_root, x_body

    def transfer(self, x, structure):
        x_root, x_body = self.split_root_body(x)

        zp, mean, logvar = self.pose_enc(x_body)
        xr = self.decoder(zp, structure)
        xr = torch.cat([x_root, xr], dim=-1)
        return xr

    def train_recon(self, x1, x2, structure, epoch):
        self.auto_encoder_optimizer.zero_grad()
        x1_root, x1_body = self.split_root_body(x1)
        x2_root, x2_body = self.split_root_body(x2)

        zp_1, mean, logvar = self.pose_enc(x1_body)
        zp_2, _, _ = self.pose_enc(x2_body)
        x1_r_body = self.decoder(zp_1, structure)
        x2_r_body = self.decoder(zp_2, structure)
        kl_loss = kl_divergence(mean, logvar).mean()

        rec_loss1 = mse_loss(x1_r_body, x1_body) 
        rec_loss2 = mse_loss(x2_r_body, x2_body) 

        reconstruction_loss = rec_loss1 + rec_loss2
        loss = reconstruction_loss + self.beta_schedule[epoch] * kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

        self.auto_encoder_optimizer.step()
        return rec_loss1, rec_loss1, kl_loss, self.beta_schedule[epoch]


    def train_generator(self, x1, x3, structure3, epoch):
        self.generator_optimizer.zero_grad()

        x1_root, x1_body = self.split_root_body(x1)
        x3_root, x3_body = self.split_root_body(x3)

        zp_1, mean, logvar = self.pose_enc(x1_body)
        xr_13 = self.decoder(zp_1, structure3)

        kl_loss = kl_divergence(mean, logvar).mean()

        # True labels
        true_labels = torch.ones(self.batch_size, 
                                 dtype=torch.long,
                                 device=x1.device)

        d1 = self.discriminator(x3_body, xr_13)

        gen_loss_1 = F.cross_entropy(d1, true_labels)

        z_random = torch.normal(0, 1, 
                                size=(self.batch_size, self.latent_dim),
                                device=x1.device)

        xr_r3 = self.decoder(z_random, structure3)
        d2 = self.discriminator(x3_body, xr_r3)
        gen_loss_2 = F.cross_entropy(d2, true_labels)

        generator_loss = gen_loss_1 + gen_loss_2 + self.beta_schedule[epoch] * kl_loss

        generator_loss.backward()

        self.generator_optimizer.step()
        return gen_loss_1, gen_loss_2, kl_loss

    def train_discriminator(self, x1, x2, x3, structure3, epoch):
        self.discriminator_optimizer.zero_grad()
        x1_root, x1_body = self.split_root_body(x1)
        x2_root, x2_body = self.split_root_body(x2)
        x2_root, x3_body = self.split_root_body(x3)

        true_labels = torch.ones(self.batch_size,
                                 dtype=torch.long, 
                                 device=x1.device)
        d_real = self.discriminator(x2_body, x3_body)
        disc_loss_real = F.cross_entropy(d_real, true_labels)

        fake_labels = torch.zeros(self.batch_size, 
                                  dtype=torch.long,
                                  device=x1.device)

        zp_1, mean, logvar = self.pose_enc(x1_body)
        xr_13 = self.decoder(zp_1, structure3)
        d_fake = self.discriminator(x3_body, xr_13)

        disc_loss_fake = F.cross_entropy(d_fake, fake_labels)


        #gp = self.gp_weight * self._gradient_penalty(self.discriminator,
        #                                             (x2_body, x3_body),
        #                                             (x2_body, xr_13))

        discriminator_loss = disc_loss_real + disc_loss_fake 
        discriminator_loss.backward()
       
        # calculate discriminator accuracy for this step
        target_true_labels = torch.cat((true_labels, fake_labels), dim=0)
       
        discriminator_predictions = torch.cat((d_real, d_fake), dim=0)
        _, discriminator_predictions = torch.max(discriminator_predictions, 1)
        discriminator_accuracy = (discriminator_predictions.data == target_true_labels.long()
                                    ).sum().item() / (self.batch_size * 2)

        if discriminator_accuracy < self.discriminator_limiting_accuracy:
            self.discriminator_optimizer.step()

        return discriminator_loss, discriminator_accuracy

    def save_model(self, path):
        model_path = os.path.join(path, 'vae_model')
        torch.save({
            "pose_encoder": self.pose_enc.state_dict(),
            "decoder": self.decoder.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }, model_path)
       
    def load_model(self, path):
        model_path = os.path.join(path, 'vae_model')
        data = torch.load(model_path)
        self.pose_enc.load_state_dict(data['pose_encoder'])
        self.decoder.load_state_dict(data['decoder'])
        self.discriminator.load_state_dict(data['discriminator'])