# TODO: Implement the dataset class extending torch.utils.data.Dataset
import pickle
import torch
import numpy as np


# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CIFAR100(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        path = f"{root}/train"
        if not train:
            path = f"{root}/test"

        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        self.data = data["data"]
        self.labels = data["fine_labels"]

        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx] / 255.0
        # TODO: Normalize as well
        label = self.labels[idx]
        return img, label
