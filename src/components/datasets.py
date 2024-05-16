from os import path
from PIL import Image
from pandas import read_csv
import numpy as np

import torch
from torch.utils.data import Dataset


class PolyMNIST(Dataset):
    def __init__(self, csv_file, transform=None, label_transform=None, return_poly=True):
        self.df = read_csv(csv_file)
        self.path = path.dirname(csv_file)
        self.transform = transform
        self.label_transform = label_transform
        self.return_poly = return_poly

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_path = path.join(self.path, self.df.file_path[index])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        if not self.return_poly:
            label = int(self.df.label[index])

            if self.label_transform:
                label = self.label_transform(label)

            return image, label

        polygon = self.df.polygon[index]
        polygon = np.fromstring(polygon, sep=",")
        polygon = polygon.reshape(-1, 3)
        polygon = torch.tensor(polygon, dtype=torch.float)

        length = int(self.df.length[index])
        pad_mask = torch.ones(19, dtype=torch.bool)
        if length < 19:
            pad_mask[length - 1:] = False

        return image, polygon, pad_mask


class PolyBean(Dataset):
    def __init__(self, csv_file, transform=None, label_transform=None, return_poly=True):
        self.df = read_csv(csv_file)
        self.path = path.dirname(csv_file)
        self.transform = transform
        self.label_transform = label_transform
        self.return_poly = return_poly

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_path = path.join(self.path, self.df.file_path[index])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        polygon = self.df.polygon[index]
        polygon = np.fromstring(polygon, sep=",")
        polygon = polygon.reshape(-1, 3)
        polygon = torch.tensor(polygon, dtype=torch.float)

        length = int(self.df.length[index])
        pad_mask = torch.ones(199, dtype=torch.bool)
        if length < 199:
            pad_mask[length - 1:] = False

        return image, polygon, pad_mask
