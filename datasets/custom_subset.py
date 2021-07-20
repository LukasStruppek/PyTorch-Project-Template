import numpy as np
import torch


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = np.array(dataset.targets)[self.indices]

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        if self.transform:
            im = self.transform(im)
        return im, targets

    def __len__(self):
        return len(self.indices)


class SingleClassSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, class_idx):
        self.dataset = dataset
        self.indices = np.where(np.array(dataset.targets) == class_idx)[0]
        self.targets = np.array(dataset.targets)[self.indices]
        self.class_idx = class_idx

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        return im, targets

    def __len__(self):
        return len(self.indices)
