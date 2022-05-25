import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import transforms as T


def prepare_data(root, save_path='./data/comma10k', size=(640, 480)):
    img_path = os.path.join(root, 'imgs')
    mask_path = os.path.join(root, 'masks')

    img_save = os.path.join(save_path, 'imgs')
    mask_save = os.path.join(save_path, 'masks')
    os.makedirs(img_save, exist_ok=True)
    os.makedirs(mask_save, exist_ok=True)

    files = os.listdir(img_path)

    for file in tqdm(files):
        img = Image.open(os.path.join(img_path, file))
        mask = Image.open(os.path.join(mask_path, file))
        img = img.resize(size, Image.LANCZOS)
        mask = mask.resize(size, Image.NEAREST)
        img.save(os.path.join(img_save, file))
        mask.save(os.path.join(mask_save, file))


class Comma10k(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, normalize=False):
        self.transform = transform
        with open('comma10k.json', 'r') as json_file:
            file_names = json.load(json_file)
        self.files = file_names['train'] if train else file_names['val']
        self.image_root = os.path.join(root, 'imgs')
        self.label_root = os.path.join(root, 'masks')
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, label = self.getdata(idx)
        image = np.asarray(image, dtype=np.float32) / 255
        if self.normalize:
            image = (image - 0.5) * 2
        label = (np.asarray(label, dtype=np.int64)[:, :, 0] + 1) >> 6
        image = torch.from_numpy(image.transpose(2, 0, 1))
        label = torch.from_numpy(label)
        return image, label

    def getdata(self, idx):
        file = self.files[idx]
        image_path = os.path.join(self.image_root, file)
        label_path = os.path.join(self.label_root, file)
        image = Image.open(image_path)
        label = Image.open(label_path).convert('RGB')
        if self.transform:
            image, label = self.transform(image, label)
        return image, label


def test_prepare_data():
    prepare_data('../comma10k', './data/comma10k_128x96', (128, 96))


def test_getdata():
    T_train = T.Compose([
        # T.RandomResize(512, 1024),
        # T.RandomCrop(512),
        T.RandomHorizontalFlip(0.5)
    ])

    comma10k = Comma10k('data/comma10k_128x96', True, T_train)
    print(len(comma10k))

    for i in range(len(comma10k)):
        image, label = comma10k.getdata(i)
        cv2.imshow('image', np.asarray(image)[:, :, ::-1])
        cv2.imshow('label', np.asarray(label)[:, :, ::-1])
        key = cv2.waitKey(0)
        if key == 27:
            break


def test_getitem():
    T_train = T.Compose([
        T.RandomResize(512, 1024),
        T.RandomCrop(512),
        T.RandomHorizontalFlip(0.5)
    ])

    comma10k = Comma10k('data/comma10k', False, T_train)
    print(len(comma10k))

    for i in range(len(comma10k)):
        image, label = comma10k[i]
        print(image.shape, label.shape)
        print(np.unique(label))
        image = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        label = label.numpy().astype(np.uint8) * 63
        cv2.imshow('image', image[:, :, ::-1])
        cv2.imshow('label', label)
        key = cv2.waitKey(0)
        if key == 27:
            break


def test_speed():
    T_train = T.Compose([
        T.RandomResize(512, 1024),
        T.RandomCrop(512),
        T.RandomHorizontalFlip(0.5)
    ])

    train_dataset = Comma10k('data/comma10k', False, T_train)
    for i in tqdm(train_dataset):
        pass


def test_dataloader():
    T_train = T.Compose([
        T.RandomResize(512, 1024),
        T.RandomCrop(512),
        T.RandomHorizontalFlip(0.5)
    ])

    train_dataset = Comma10k('data/comma10k', False, T_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)
    for i in tqdm(train_loader):
        pass


if __name__ == '__main__':
    # test_prepare_data()
    test_getdata()
    # test_getitem()
    # test_speed()
    # test_dataloader()
