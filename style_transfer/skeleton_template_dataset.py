import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class SkeletonTemplateDataset(Dataset):
    """replay buffer dataset."""

    def __init__(self, sizes):
        """
        Args:
            data_file (string): Path to the data file.
        """
        self._data = sizes
        self.N = len(sizes)
        self.max_size = max(sizes)

    def __len__(self):
        return 1280

    def __getitem__(self, idx):
        i = torch.randint(self.N, (1,))
        s = self._data[i]
        sample = torch.zeros(self.max_size)
        sample[:s] = 1.
        return sample