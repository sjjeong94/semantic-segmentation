import os
import time
import logging
import random
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

import datasets
import transforms


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train(
    logs_root,
    data_root='data/comma10k',
    learning_rate=0.0003,
    weight_decay=0,
    batch_size=8,
    epochs=50,
    num_workers=2,
):
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(
        os.path.join(logs_root, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = 'mobilenet_v2'  # 'efficientnet-b0'
    net = smp.Unet(backbone, classes=5)
    net = net.to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    model_files = sorted(os.listdir(model_path))
    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']

    T_train = transforms.Compose([
        transforms.RandomResize(512, 1024),
        transforms.RandomCrop(size=512),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    T_val = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.Comma10k(data_root, True, T_train)
    val_dataset = datasets.Comma10k(data_root, False, T_val)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    criterion = nn.CrossEntropyLoss()

    # logger.info(net)
    # logger.info(device)
    logger.info(optimizer)

    logger.info('| %12s | %12s | %12s | %12s | %12s |' %
                ('epoch', 'time_train', 'time_val', 'loss_train', 'loss_val'))

    for epoch in range(epoch_begin, epochs):
        t0 = time.time()
        net.train()
        losses = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = net(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            losses += loss.detach()

        loss_train = losses / len(train_loader)
        t1 = time.time()
        time_train = t1 - t0

        t0 = time.time()
        net.eval()
        losses = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x = x.to(device)
                y = y.to(device)

                out = net(x)
                loss = criterion(out, y)

                losses += loss.detach()

        loss_val = losses / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        logger.info('| %12d | %12.4f | %12.4f | %12.4f | %12.4f |' %
                    (epoch + 1, time_train, time_val, loss_train, loss_val))

        model_file = os.path.join(model_path, 'model_%03d.pt' % (epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_val.item(),
        }, model_file)


if __name__ == '__main__':
    train(
        logs_root='logs/comma10k/test3',
        epochs=2,
        learning_rate=0.0001,
    )
