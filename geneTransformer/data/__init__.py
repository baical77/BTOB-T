import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from logger import logging
from torchvision.datasets.folder import DatasetFolder


def custom_loader(path):
    ret = torch.load(path)
    return ret

def get_dataloaders(
        train_dir,
        val_dir,
        split=(0.5, 0.5),
        batch_size=32,
        *args, **kwargs):
    """
    This function returns the train, val and test dataloaders.
    """
    # create the datasets
    train_ds = DatasetFolder(root=train_dir, loader=custom_loader, extensions='.pt')
    val_ds = DatasetFolder(root=val_dir, loader=custom_loader, extensions='.pt')
    # now we want to split the val_ds in validation and test
    lengths = np.array(split) * len(val_ds)
    lengths = lengths.astype(int)
    left = len(val_ds) - lengths.sum()
    # we need to add the different due to float approx to int
    lengths[-1] += left

    val_ds, test_ds = random_split(val_ds, lengths.tolist())
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl
