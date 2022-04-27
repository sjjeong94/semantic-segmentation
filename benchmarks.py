import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, fcn_resnet50, deeplabv3_mobilenet_v3_large
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


def benchmarks(
    root,
    net,
    learning_rate=0.0003,
    weight_decay=0,
    batch_size=2,
    epochs=1,
    normalize=False,
    random_crop=False,
    random_flip=False,
    num_workers=4,
    pin_memory=True,
):
    set_seed(1234)
    os.makedirs(root, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    T_train = []
    if random_crop:
        T_train.extend([T.RandomResize(512, 1024), T.RandomCrop(512)])
    if random_flip:
        T_train.append(T.RandomHorizontalFlip(0.5))
    T_train = T.Compose(T_train)
    print(T_train)

    train_dataset = Comma10k(
        '../comma10k', True, normalize=normalize, transform=T_train)
    val_dataset = Comma10k('../comma10k', False, normalize=normalize)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(net)
    print(device)
    print(optimizer)

    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        t0 = time.time()
        net.train()
        losses = 0
        for i, (x, y) in enumerate(tqdm(train_loader)):
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
        duration = t1 - t0
        print('loss train : ', loss_train)

        net.eval()
        losses = 0
        for i, (x, y) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)

                out = net(x)['out']
                loss = criterion(out, y)

                losses += loss.detach()

        loss_val = losses / len(val_loader)
        print('loss val : ', loss_val)

        print('| epoch %3d | %12.4f | %12.4f | %12.4f |' %
              (epoch, duration, loss_train, loss_val))

        model_path = os.path.join(root, 'model_%03d.pth' % epoch)
        torch.save(net.state_dict(), model_path)


if __name__ == '__main__':
    #benchmarks('results/000', lraspp_mobilenet_v3_large(num_classes=5))
    #benchmarks('results/001', fcn_resnet50(num_classes=5))
    # benchmarks('results/002',
    #          deeplabv3_mobilenet_v3_large(num_classes=5), batch_size=4)
    # benchmarks('results/002_1',
    #           deeplabv3_mobilenet_v3_large(num_classes=5), batch_size=8)
    # benchmarks('results/002_2',
    #           deeplabv3_mobilenet_v3_large(num_classes=5), batch_size=8, normalize=True)
    # benchmarks('results/002_3',
    #           deeplabv3_mobilenet_v3_large(num_classes=5), batch_size=8, epochs=5, normalize=True)
    # benchmarks('results/002_4',
    #           deeplabv3_mobilenet_v3_large(num_classes=5), batch_size=8, epochs=5, normalize=True, random_flip=True)
    # benchmarks('results/002_5',
    #           deeplabv3_mobilenet_v3_large(num_classes=5), batch_size=8, epochs=5, normalize=True, random_crop=True)
    benchmarks('results/002_5',
               deeplabv3_mobilenet_v3_large(num_classes=5), batch_size=8, epochs=30, normalize=True, random_crop=True)
