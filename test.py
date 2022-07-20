import cv2
import torch

from datasets import Comma10k, Cityscapes
import transforms


def test_dataset():
    T_compose = transforms.Compose([
        # transforms.RandomResize(512, 1024),
        # transforms.RandomCrop(size=512),
        transforms.Resize((640, 480)),
        transforms.RandomHorizontalFlip(),
    ])
    dataset = Comma10k('./data/comma10k', transform=T_compose)

    idx = 0
    while True:
        print('%d / %d' % (idx, len(dataset)))

        image, label = dataset[idx]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        label = cv2.cvtColor(dataset.get_color(label), cv2.COLOR_RGB2BGR)

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

    cv2.destroyAllWindows()


def test_cityscape():

    T_compose = transforms.Compose([
        transforms.RandomResize(512, 1024),
        transforms.RandomCrop(size=512),
        #transforms.Resize((512, 256)),
        # transforms.RandomHorizontalFlip(),
    ])

    dataset = Cityscapes('./data/cityscapes', 'train', T_compose)

    idx = 0
    while True:
        print('%d / %d' % (idx, len(dataset)))
        image, label = dataset[idx]

        label = dataset.get_color(label)
        cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imshow('label', cv2.cvtColor(label, cv2.COLOR_RGB2BGR))

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

    cv2.destroyAllWindows()


def test_dataset2():
    T_compose = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResize(512, 1024),
        transforms.RandomCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0, 0.5), (0.5, 0.5, 1))
    ])
    dataset = Comma10k('./data/comma10k', transform=T_compose)

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
    dataset = Comma10k('./data/comma10k', transform=T_compose)

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


if __name__ == '__main__':
    test_dataset()
    test_cityscape()
    test_dataset2()
    test_dataloader()
