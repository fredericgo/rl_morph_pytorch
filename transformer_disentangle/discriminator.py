import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(
        self,
        root_size,
        feature_size,
        max_num_limbs,
        dropout=0.5,):
        super(Discriminator, self).__init__()

        self.model_type = "Discriminator"
        self.root_size = root_size
        self.max_num_limbs = max_num_limbs
        self.state_size = feature_size
        self.max_state_dim = self.root_size + self.state_size * (self.max_num_limbs - 1)

        self.mlp = nn.Sequential(
            nn.Linear(self.max_state_dim*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):     
        x = torch.cat([x1, x2], dim=1)
        return self.mlp(x)


if __name__ == "__main__":
    disc = Discriminator(
                root_size=11,
                feature_size=2,
                max_num_limbs=13
           )

    x1 = torch.randn(3, 35)
    x2 = torch.randn(3, 35)

    s = disc(x1, x2)
    print(s)