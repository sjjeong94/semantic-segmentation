import cv2
import torch
import torchvision
import videocv
import numpy as np
import segmentation_models_pytorch as smp

import transforms
import datasets


def test_dataset():
    T_compose = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResize(512, 1024),
        transforms.RandomCrop(size=512),
    ])
    dataset = datasets.Comma10k('./data/comma10k', transform=T_compose)

    idx = 0
    while True:
        print('%d / %d' % (idx, len(dataset)))

        image, label = dataset[idx]
        print(label.dtype)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        label = cv2.cvtColor(datasets.label_decode(label), cv2.COLOR_RGB2BGR)

        cv2.imshow('image', image)
        cv2.imshow('label', label)

        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('q'):
            idx -= 1
        else:
            idx += 1

        if idx < 0:
            idx = len(dataset)-1
        elif idx >= len(dataset):
            idx = 0


def test_dataset2():
    T_compose = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResize(512, 1024),
        transforms.RandomCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0, 0.5), (0.5, 0.5, 1))
    ])
    dataset = datasets.Comma10k('./data/comma10k', transform=T_compose)

    image, label = dataset[4]
    print(image.shape, image.dtype, image.max(), image.min())
    print(label.shape, label.dtype, torch.unique(label))
    print(image[0].min(), image[0].max())
    print(image[1].min(), image[1].max())
    print(image[2].min(), image[2].max())


def test_dataloader():
    T_compose = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResize(512, 1024),
        transforms.RandomCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0, 0.5), (0.5, 0.5, 1))
    ])
    dataset = datasets.Comma10k('./data/comma10k', transform=T_compose)

    batch_size = 8

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    for x, y in loader:
        break

    print('DataLoader Test')
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)


def test_model():
    backbone = 'efficientnet-b0'
    net = smp.Unet(backbone, classes=5)
    x = torch.randn((8, 3, 512, 512))
    out = net(x)
    print(out.shape)


if __name__ == '__main__':
    test_dataset()
    test_dataset2()
    test_dataloader()
    test_model()
