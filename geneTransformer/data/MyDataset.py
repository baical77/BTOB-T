import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.x_data = np.load(root_dir / 'input.npy')
        self.y_data = np.load(root_dir / 'label.npy')
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x': torch.from_numpy(self.x_data[idx]), 'y': torch.from_numpy(self.y_data[idx])}

        return sample

    def __len__(self):
        return len(self.y_data)