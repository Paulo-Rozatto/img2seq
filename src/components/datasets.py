from os import path
from PIL import Image
from pandas import read_csv
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip, vflip

from matplotlib import pyplot as plt

class PolyMNIST(Dataset):
    def __init__(self, csv_file, transform=None, label_transform=None, return_poly=True):
        df = read_csv(csv_file)
        dir_path = path.dirname(csv_file)

        df.label = df.label.astype(int)

        self.data = []
        for _, entry in df.iterrows():
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
        for _, entry in df.iterrows():
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


class AugmentBean(Dataset):
    def hflip(self, image, polygon, pad_mask):
        flip_image = hflip(image)
        flip_polygon = polygon.clone()
        flip_polygon[1:, 1] = 1 - flip_polygon[1:, 1]

        return flip_image, flip_polygon, pad_mask
        # self.data.append((flip_image, flip_polygon, pad_mask))
    
    def vflip(self, image, polygon, pad_mask):
        flip_image = vflip(image)
        flip_polygon = polygon.clone()
        flip_polygon[1:, 2] = 1 - flip_polygon[1:, 2]

        return flip_image, flip_polygon, pad_mask
        # self.data.append((flip_image, flip_polygon, pad_mask))

    def __init__(self, csv_file, transform=None, label_transform=None, test=False):
        df = read_csv(csv_file)
        dir_path = path.dirname(csv_file)
        self.test = test

        self.data = []
        for _, entry in df.iterrows():
            image_path = path.join(dir_path, entry.file_path)

            image = Image.open(image_path)

            if transform:
                image = transform(image)

            polygon = np.fromstring(entry.polygon, sep=",")
            polygon = polygon.reshape(-1, 2)[:199]
            polygon = np.c_[
                np.arange(1, len(polygon) + 1) / (len(polygon)), polygon]
            polygon = np.r_[np.array([[0.0, 0.0, 0.0]]), polygon]

            length = len(polygon)
            end = np.array([polygon[0]])
            # end = np.array([[1.0, 0.0, 0.0]])
            for i in range(0, 200 - length):
                polygon = np.r_[polygon, end]

            polygon = torch.tensor(polygon, dtype=torch.float)

            pad_mask = torch.ones(199, dtype=torch.bool)

            if length < 199:
                pad_mask[length:] = False

            self.data.append((image, polygon, pad_mask))

        # length = len(self.data)
        # for i in range(length):
        #     image, polygon, pad_mask = self.data[i]
        #     self.hflip(image, polygon, pad_mask)

        # for image, polygon, pad_mask in self.data:
        #     self.hflip(image, polygon, pad_mask)

        # for image, polygon, pad_mask in self.data:
        #     self.vflip(image, polygon, pad_mask)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.test:
             return self.data[index]
        n = torch.rand(1)[0]
        if n <= 0.33:
            return self.data[index]
        elif n <= 0.66:
            image, polygon, pad_mask = self.data[index]
            return self.hflip(image, polygon, pad_mask)
        else:
            image, polygon, pad_mask = self.data[index]
            return self.vflip(image, polygon, pad_mask)
