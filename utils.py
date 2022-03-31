import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import transforms as T
from PIL import Image
from tqdm import tqdm
from dataset import Comma10k

mapping_20 = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 2,
    9: 0,
    10: 0,
    11: 3,
    12: 4,
    13: 5,
    14: 0,
    15: 0,
    16: 0,
    17: 6,
    18: 0,
    19: 7,
    20: 8,
    21: 9,
    22: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
    29: 0,
    30: 0,
    31: 17,
    32: 18,
    33: 19,
    -1: 0
}


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
    def __init__(self, root, split, mode, target_type, transform=None):
        super().__init__(root, split, mode, target_type)
        self.transform = transform

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        if self.transform:
            image, target = self.transform(image, target)
        image = np.asarray(image, dtype=np.float32) / 255
        target = np.asarray(target)

        # convert class number
        label = np.zeros_like(target, dtype=np.int64)
        for k in mapping_20:
            label[target == k] = mapping_20[k]

        image = torch.from_numpy(image.transpose(2, 0, 1))
        label = torch.from_numpy(label)
        return image, label


def get_cityscapes(root, batch_size):

    T_train = T.Compose([
        T.RandomResize(512, 2048),
        T.RandomCrop(512),
        T.RandomHorizontalFlip(0.5)
    ])

    train_dataset = Cityscapes(
        root, 'train', 'fine', 'semantic', transform=T_train)

    val_dataset = Cityscapes(
        root, 'val', 'fine', 'semantic')

    print(train_dataset)
    print(val_dataset)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_comm10k(root, batch_size):
    T_train = T.Compose([
        T.RandomResize(512, 1024),
        T.RandomCrop(512),
        T.RandomHorizontalFlip(0.5)
    ])

    train_dataset = Comma10k('../comma10k', True, T_train)
    val_dataset = Comma10k('../comma10k', False)

    print(train_dataset)
    print(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_miou(y, o, num_classes):
    m = y == o
    iou_list = []
    for i in range(1, num_classes):
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
            x = x.to(self.device)
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
                x = x.to(self.device)
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
        for _, (x, y) in enumerate(tqdm(self.val_loader)):
            with torch.no_grad():
                lab = y.numpy()
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.net(x)['out']
                pred = torch.argmax(out, 1)
                pred = pred.cpu().numpy()
                for i in range(len(lab)):
                    miou = get_miou(lab[i], pred[i], 20)
                    miou_list.append(miou)
                del x, y, out, pred, lab
                torch.cuda.empty_cache()
        miou = np.mean(miou_list)
        print('mIoU = %.3f' % (miou * 100))


colormap = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (244, 35, 232),
    3: (70, 70, 70),
    4: (102, 102, 156),
    5: (190, 153, 153),
    6: (153, 153, 153),
    7: (250, 170, 30),
    8: (220, 220, 0),
    9: (107, 142, 35),
    10: (152, 251, 152),
    11: (70, 130, 180),
    12: (220, 20, 60),
    13: (255, 0, 0),
    14: (0, 0, 142),
    15: (0, 0, 70),
    16: (0, 60, 100),
    17: (0, 80, 100),
    18: (0, 0, 230),
    19: (119, 11, 32)
}


class CityscapesDemo:
    def __init__(self, root, net):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        net.eval()
        net.to(self.device)
        self.net = net

        dirs = ['stuttgart_00', 'stuttgart_01', 'stuttgart_02']
        paths = []
        for d in dirs:
            path0 = os.path.join(root, 'leftImg8bit/demoVideo', d)
            files = sorted(os.listdir(path0))
            for f in files:
                if f[-3:] == 'png':
                    path = os.path.join(path0, f)
                    paths.append(path)
        self.paths = paths

        self.inference()

    def inference(self):
        preds = []
        for path in tqdm(self.paths):
            image = Image.open(path)

            x = np.array(image, np.float32).transpose(2, 0, 1) / 255
            x = torch.from_numpy(x).unsqueeze(0)

            with torch.no_grad():
                x = x.to(self.device)
                out = self.net(x)['out']
                pred = torch.argmax(out, 1)
                pred = pred.cpu().numpy()[0]

            pred = pred.astype(np.uint8)
            preds.append(pred)

        self.preds = preds

    def get_colorview(self, pred):
        H, W = pred.shape
        view = np.zeros((H, W, 3), dtype=np.uint8)
        for k in colormap.keys():
            mask = k == pred
            view[mask] = colormap[k]
        view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        return view

    def make_video(self, video_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        w, h = 1920, 1080
        video = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for i in tqdm(range(len(self.preds))):
            image = cv2.imread(self.paths[i])
            pred = self.preds[i]
            view = self.get_colorview(pred)
            overlap = (image * 0.666 + view * 0.333).astype(np.uint8)
            overlap = cv2.resize(overlap, (w, h))
            video.write(overlap)

        video.release()
