import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
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


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    return device


class Cityscapes(torchvision.datasets.Cityscapes):
    def __init__(self, root, split, mode, target_type, transform):
        super().__init__(root, split, mode, target_type)
        self.transform = transform

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        if self.transform:
            image, target = self.transform(image, target)
        return image, target


def get_cityscapes(root, batch_size):

    T_train = T.Compose([
        T.RandomResize(512, 2048),
        T.RandomCrop(512),
        T.RandomHorizontalFlip(0.5),
        T.PILToTensor()
    ])

    T_val = T.Compose([
        T.PILToTensor()
    ])

    train_dataset = Cityscapes(
        root, 'train', 'fine', 'semantic', transform=T_train)

    val_dataset = Cityscapes(
        root, 'val', 'fine', 'semantic', transform=T_val)

    print(train_dataset)
    print(val_dataset)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_miou(y, o, num_classes):
    m = y == o
    iou_list = []
    for i in range(num_classes):
        myi = m[y == i]
        moi = m[o == i]
        inter = myi.sum()
        union = len(myi) + len(moi) - inter
        iou = inter / union
        iou_list.append(iou)
    miou = np.nanmean(iou_list)
    return miou


class Engine:
    def __init__(
        self,
        name,
        net,
        device,
        optimizer,
        train_loader,
        val_loader,
        model_root,
        result_root,
        image_root,
    ):
        self.name = name
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_root = model_root
        self.result_root = result_root
        self.image_root = image_root

    def train_one_epoch(self, use_amp):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        criterion = nn.CrossEntropyLoss()
        self.net.train()
        losses = []
        for i, (x, y) in enumerate(tqdm(self.train_loader)):
            x = x.to(self.device) / 255.  # normalize
            y = y.to(self.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = self.net(x)['out']
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            losses.append(loss.cpu().item())
        loss_train = np.mean(losses)
        return loss_train

    def evaluate_one_epoch(self):

        criterion = nn.CrossEntropyLoss()
        self.net.eval()
        losses = []
        for i, (x, y) in enumerate(tqdm(self.val_loader)):
            with torch.no_grad():
                x = x.to(self.device) / 255.  # normalize
                y = y.to(self.device)

                out = self.net(x)['out']
                loss = criterion(out, y)

                losses.append(loss.cpu().item())
        loss_val = np.mean(losses)
        return loss_val

    def train(self, epoch_begin=0, epoch_end=100, use_amp=True):
        os.makedirs(self.model_root, exist_ok=True)
        os.makedirs(self.result_root, exist_ok=True)

        losses_train = []
        for epoch in range(epoch_begin, epoch_end+1):

            loss_train = self.train_one_epoch(use_amp)

            model_path = os.path.join(
                self.model_root, '%s_epoch%03d.pth' % (self.name, epoch))
            torch.save(self.net.state_dict(), model_path)

            losses_train.append(loss_train)

            print('epoch %4d  |  loss %9.6f  |   model_path -> %s' %
                  (epoch, loss_train, model_path))

        result = np.array(losses_train)

        result_path = os.path.join(self.result_root, '%s.npy' % self.name)
        np.save(result_path, result)
        print('result_path:', result_path)

        self.result = result

    def evaluate(self):
        self.net = self.net.eval()
        miou_list = []
        with torch.no_grad():
            for _, (x, y) in enumerate(tqdm(self.val_loader)):
                lab = y.numpy()
                x = x.to(self.device) / 255.
                y = y.to(self.device)
                out = self.net(x)['out']
                pred = torch.argmax(out, 1)
                pred = pred.cpu().numpy()
                for i in range(len(lab)):
                    miou = get_miou(lab[i], pred[i], 34)
                    miou_list.append(miou)
        miou = np.mean(miou_list)
        print('mIoU = %.3f' % (miou * 100))
