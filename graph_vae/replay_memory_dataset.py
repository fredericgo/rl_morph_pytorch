import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class ReplayMemoryDataset(Dataset):
    """replay buffer dataset."""

    def __init__(self, data_file, skeleton):
        """
        Args:
            data_file (string): Path to the data file.
        """
        self._data = torch.load(data_file)
        self._skeleton = skeleton

    def __len__(self):
        return len(self._data['reward'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [torch.tensor(v[idx]) for v in self._data.values()] + [self._skeleton]
        return sample