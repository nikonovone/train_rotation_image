import numpy as np
import torch
from torch.utils.data import Dataset


class RotateDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_image = torch.zeros(1, 64, 64)
        x_start = np.random.randint(0, 64 - 16)
        y_start = np.random.randint(0, 64 - 16)
        input_image[0, y_start : y_start + 16, x_start : x_start + 16] = 1.0

        target_image = torch.rot90(input_image, k=1, dims=(1, 2))

        return input_image, target_image
