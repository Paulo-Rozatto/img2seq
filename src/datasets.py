from os import path
from PIL import Image
# from pandas import read_csv
import pandas as pd
import numpy as np
# from numpy import fromstring, zeros

import torch
from torch.utils.data import Dataset

class PolyMNIST(Dataset):
    def __init__(self, csv_file, transform=None, label_transform=None, return_poly=True):
        self.df = pd.read_csv(csv_file)
        self.path = path.dirname(csv_file)
        self.transform = transform
        self.label_transform = label_transform
        self.return_poly = return_poly

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_path = path.join(self.path, self.df.file_path[index])
        image = Image.open(image_path)

        label = int(self.df.label[index])

        if self.transform:
            image = self.transform(image)

        if self.label_transform:
            label = self.label_transform(label)

        if not self.return_poly:
            return image, label

        polygon = self.df.polygon[index]
        polygon = np.fromstring(polygon, sep=",")

        polygon = polygon.reshape(-1, 2)
        polygon = np.c_[np.arange(1, len(polygon) + 1) /
                        (len(polygon) + 1), polygon]
        y = np.array([[0, 0, 0]])
        z = np.array([[1, 0, 0]])
        polygon = np.r_[y, polygon, z]

        for i in range(0, 12 - len(polygon)):
            polygon = np.r_[polygon, z]

        polygon = torch.tensor(polygon, dtype=torch.float)

        return image, label, polygon
