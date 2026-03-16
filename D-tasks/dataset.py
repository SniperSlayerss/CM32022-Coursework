import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import numpy as np

import os
import dataset
from typing import NamedTuple, Optional


# Calculated only on training set "/data/train.pkl"
CIFAR_100_MEAN = np.array([0.5052151083946228, 0.4850522577762604, 0.4412228465080261], dtype=np.float32)
CIFAR_100_STD = np.array([0.26796528697013855, 0.25680774450302124, 0.27619215846061707], dtype=np.float32)


# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CIFAR100(Dataset):
    def __init__(self, root: str, train: bool):
        self.root = root
        self.train = train

        path = f"{root}/train.pkl"
        if not train:
            path = f"{root}/test.pkl"

        with open(path, "rb") as f:
            data = pickle.load(f, encoding="bytes")

        self.data = data[b"data"]
        self.fine_labels = torch.tensor(data[b"fine_labels"], dtype=torch.long)
        self.coarse_labels = torch.tensor(data[b"coarse_labels"], dtype=torch.long)

        # self.data = torch.tensor(
        #     self.data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0,
        #     dtype=torch.float32,
        # )

        self.data = torch.tensor(
            (self.data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0 - CIFAR_100_MEAN.reshape(1, 3, 1, 1)) / CIFAR_100_STD.reshape(1, 3, 1, 1),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        # img = self.data[idx]

        if self.train:
            # Horizontal flip
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[2])

            # Random crop
            pad = 4
            img = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode="reflect")

            i = torch.randint(0, 2 * pad + 1, (1,)).item()
            j = torch.randint(0, 2 * pad + 1, (1,)).item()
            img = img[:, i : i + 32, j : j + 32]

            # Brightness
            if torch.rand(1) > 0.5:
                brightness = 1 + (torch.rand(1).item() * 0.3 - 0.15)
                img = torch.clamp(img * brightness, -3, 3)

            # Random erasing
            if torch.rand(1) > 0.75:
                erase = torch.randint(6, 12, (1,)).item()
                x = torch.randint(0, 32 - erase, (1,)).item()
                y = torch.randint(0, 32 - erase, (1,)).item()
                img[:, x : x + erase, y : y + erase] = 0

        return img, {"fine": self.fine_labels[idx], "coarse": self.coarse_labels[idx]}


def get_training_stats():
    with open(f"{os.path.dirname(__file__)}/data/train.pkl", "rb") as f:
        data = pickle.load(f, encoding="bytes")

    imgs = data[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    mean = imgs.mean(axis=(0, 2, 3))
    std = imgs.std(axis=(0, 2, 3))
    print(len(imgs))

    print(f"CIFAR_100_MEAN = np.array({mean.tolist()}, dtype=np.float32)")
    print(f"CIFAR_100_STD = np.array({std.tolist()}, dtype=np.float32)")


class Data(NamedTuple):
    train_set: Dataset
    test_set: Dataset
    train_dataloader: DataLoader
    test_dataloader: DataLoader


def init_dataloaders(batch_size=32, sampler=None) -> Data:
    # Prepare data
    train_set = dataset.CIFAR100(
        f"{os.path.dirname(__file__)}/data",
        train=True,
    )
    test_set = dataset.CIFAR100(f"{os.path.dirname(__file__)}/data", train=False)

    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        persistent_workers=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)

    return Data(
        train_set=train_set,
        test_set=test_set,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )


if __name__ == "__main__":
    get_training_stats()
