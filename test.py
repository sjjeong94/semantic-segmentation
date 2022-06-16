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


def test_inference():
    model_path = './logs/comma10k/test/models_/model_005.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = 'mobilenet_v2'  # 'efficientnet-b0'
    net = smp.Unet(backbone, classes=5)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net = net.eval()

    T_Compose = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    size = (1024, 576)
    video = videocv.Video('./videos/test.mp4')

    idx = 0
    while video():
        idx += 1
        print('%d / %d' % (idx, video.frame_count))

        image = cv2.resize(video.frame, size, interpolation=cv2.INTER_AREA)

        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = T_Compose(x).unsqueeze(0)
        x = x.to(device)
        with torch.inference_mode():
            out = net(x)
            y = torch.argmax(out, 1)
            y = y.cpu().numpy().squeeze()
        view = datasets.label_decode(y)
        view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

        cv2.imshow('image', image)
        cv2.imshow('view', view)


if __name__ == '__main__':
    # test_dataset()
    # test_dataset2()
    # test_dataloader()
    # test_model()
    test_inference()
