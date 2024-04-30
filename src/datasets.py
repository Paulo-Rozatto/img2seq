from os import path
from PIL import Image
from pandas import read_csv
from numpy import fromstring

import torch
from torch.utils.data import Dataset

class PolyMNIST(Dataset):
    def __init__(self, csv_file, transform=None, label_transform=None):
        self.df = read_csv(csv_file)
        self.path = path.dirname(csv_file)
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_path = path.join(self.path, self.df.file_path[index])
        image = Image.open(image_path)

        label = int(self.df.label[index])

        polygon = self.df.polygon[index]
        polygon = torch.tensor(
            fromstring(polygon, sep=",").reshape(-1, 2)
        )

        if self.transform:
            image = self.transform(image)

        if self.label_transform:
            label = self.label_transform(label)

        return image, label
