# TODO: Implement the dataset class extending torch.utils.data.Dataset
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os
import dataset
from typing import NamedTuple


# TODO: Calculate for given dataset, check if this makes sense to do with new test data
# TODO: Use full CIFAR test set to check if it generalises
# CIFAR_100_MEAN = np.array([0.5073625, 0.4866923, 0.44109058], dtype = np.float32)
# CIFAR_100_STD = np.array([0.26748824, 0.25659314, 0.2763088], dtype = np.float32)
# TODO: Potentially handle using batch norm instead?
# TODO: Download dataset?

# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CIFAR100(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        path = f"{root}/train.pkl"
        if not train:
            path = f"{root}/test.pkl"

        with open(path, "rb") as f:
            data = pickle.load(f, encoding="bytes")

        # TODO: Make sure the b is needed
        self.data = data[b"data"]
        self.fine_labels = data[b"fine_labels"]
        self.coarse_labels = data[b"coarse_labels"]

        self.data = self.data.reshape(-1, 3, 32, 32).astype(np.float32)

        # self.mean = CIFAR_100_MEAN.reshape(3, 1, 1)
        # self.std = CIFAR_100_STD.reshape(3, 1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx] / 255.0

        # img = (img - self.mean) / self.std
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, {
            "fine": self.fine_labels[idx],
            "coarse": self.coarse_labels[idx]
        }

class Data(NamedTuple):
    train_set: Dataset
    test_set: Dataset
    train_dataloader: DataLoader
    test_dataloader: DataLoader


def init_dataloaders() -> Data:
    # Prepare data
    train_set = dataset.CIFAR100(
        f"{os.path.dirname(__file__)}/data", train=True
    )
    test_set = dataset.CIFAR100(
        f"{os.path.dirname(__file__)}/data", train=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=32, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    return Data(
        train_set=train_set,
        test_set=test_set,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

