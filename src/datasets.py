from os import path
from PIL import Image
from pandas import read_csv
from numpy import fromstring
from torch import Dataset


class PolyMNIST(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = read_csv(csv_file)
        self.path = path.dirname(csv_file)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_path = path.join(self.path, self.df.file_path[index])
        image = Image.open(image_path)
        label = self.df.label[index]
        polygon = self.df.polygon[index]
        polygon = fromstring(polygon[1:-1]).reshape(-1, 2)

        if self.transform:
            image = self.transform(image)
        return image, label
