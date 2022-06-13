import os
import time
import logging
import random
import torch
import torch.nn as nn
import numpy as np
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from dataset import Comma10k
import transforms as T
from tqdm import tqdm


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
    net=deeplabv3_mobilenet_v3_large(num_classes=5),
    data_root='data/comma10k_128x96',
    learning_rate=0.0003,
    weight_decay=0,
    batch_size=16,
    epochs=1,
    normalize=False,
    random_crop=False,
    random_flip=False,
    num_workers=2,
    pin_memory=True,
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
    net = net.to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    T_train = []
    if random_crop:
        T_train.extend([T.RandomResize(96, 192), T.RandomCrop(96)])
    if random_flip:
        T_train.append(T.RandomHorizontalFlip(0.5))
    T_train = T.Compose(T_train)
    print(T_train)

    train_dataset = Comma10k(
        data_root, True, normalize=normalize, transform=T_train)
    val_dataset = Comma10k(data_root, False, normalize=normalize)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    logger.info(net)
    logger.info(device)
    logger.info(optimizer)

    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    logger.info('| %12s | %12s | %12s | %12s | %12s |' %
                ('epoch', 'time_train', 'time_val', 'loss_train', 'loss_val'))

    for epoch in range(epochs):
        t0 = time.time()
        net.train()
        losses = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = net(x)['out']
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses += loss.detach()
        loss_train = losses / len(train_loader)
        t1 = time.time()
        time_train = t1 - t0

        t0 = time.time()
        net.eval()
        losses = 0
        for x, y in tqdm(val_loader):
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)

                out = net(x)['out']
                loss = criterion(out, y)

                losses += loss.detach()
        loss_val = losses / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        logger.info('| %12d | %12.4f | %12.4f | %12.4f | %12.4f |' %
                    (epoch + 1, time_train, time_val, loss_train, loss_val))

        model_file = os.path.join(model_path, 'model_%03d.pth' % (epoch + 1))
        torch.save(net.state_dict(), model_file)


if __name__ == '__main__':
    train('logs/comma10k/test_128x96_1', epochs=50, random_flip=True)
