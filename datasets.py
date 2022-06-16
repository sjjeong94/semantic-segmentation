import os
import json
import torch
import numpy as np
from PIL import Image


class_colors = np.asarray([
    (0x00, 0xff, 0x66),  # movable
    (0x40, 0x20, 0x20),  # road
    (0x80, 0x80, 0x60),  # undrivable
    (0xcc, 0x00, 0xff),  # my car
    (0xff, 0x00, 0x00),  # lane markings
], dtype=np.uint8)


def label_encode(label):
    return (label[:, :, 0].astype(np.int64) + 1) >> 6


def label_decode(label):
    return class_colors[label]


class Comma10k(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        with open('comma10k.json', 'r') as json_file:
            file_names = json.load(json_file)
        self.files = file_names['train'] if train else file_names['val']
        self.image_root = os.path.join(root, 'imgs')
        self.label_root = os.path.join(root, 'masks')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, label = self.getdata(idx)
        if self.transform:
            image, label = self.transform(image, label)
        return image, label

    def getdata(self, idx):
        file = self.files[idx]
        image_path = os.path.join(self.image_root, file)
        label_path = os.path.join(self.label_root, file)
        image = Image.open(image_path)
        label = Image.open(label_path).convert('RGB')
        image = np.asarray(image)
        label = np.asarray(label)
        label = label_encode(label)
        return image, label
