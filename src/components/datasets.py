from os import path
from PIL import Image
from pandas import read_csv
import numpy as np

import torch
from torch.utils.data import Dataset


class PolyMNIST(Dataset):
    def __init__(self, csv_file, transform=None, label_transform=None, return_poly=True):
        df = read_csv(csv_file)
        dir_path = path.dirname(csv_file)

        df.label = df.label.astype(int)

        self.data = []
        for _, entry  in df.iterrows():
            image_path = path.join(dir_path, entry.file_path)

            image = Image.open(image_path)
            label = entry.label

            if transform:
                image = transform(image)
            
            if label_transform:
                label = label_transform(label)

            if not return_poly:
                self.data.append((image, label))
                continue

            polygon = np.fromstring(entry.polygon, sep=",")
            polygon = polygon.reshape(-1, 3)
            polygon = torch.tensor(polygon, dtype=torch.float)

            length = int(entry.length)
            pad_mask = torch.ones(19, dtype=torch.bool)

            if length < 19:
                pad_mask[length - 1:] = False
            
            self.data.append((image, polygon, pad_mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class PolyBean(Dataset):
    def __init__(self, csv_file, transform=None, label_transform=None):
        df = read_csv(csv_file)
        dir_path = path.dirname(csv_file)

        self.data = []
        for _, entry  in df.iterrows():
            image_path = path.join(dir_path, entry.file_path)

            image = Image.open(image_path)

            if transform:
                image = transform(image)
            
            polygon = np.fromstring(entry.polygon, sep=",")
            polygon = polygon.reshape(-1, 3)
            polygon = torch.tensor(polygon, dtype=torch.float)

            length = int(entry.length)
            pad_mask = torch.ones(199, dtype=torch.bool)

            if length < 199:
                pad_mask[length - 1:] = False
            
            self.data.append((image, polygon, pad_mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
