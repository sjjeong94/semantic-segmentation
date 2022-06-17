import cv2
import numpy as np
import videocv
import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
from tqdm import tqdm

import datasets
import transforms
import metrics


class Module:
    def __init__(self, model_path):

        device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        self.device = device
        self.net = net
        self.transform = T_Compose

    @torch.inference_mode()
    def __call__(self, image):
        x = self.transform(image).unsqueeze(0)
        x = x.to(self.device)
        out = self.net(x)
        y = torch.argmax(out, 1)
        y = y.cpu().numpy().squeeze()
        return y


def evaluate(
    data_root='./data/comma10k',
    model_path='./logs/comma10k/test1/models/model_050.pth',
    size=(640, 480),
):
    module = Module(model_path)

    dataset = datasets.Comma10k(
        data_root,
        False,
        transforms.Resize(size)
    )

    results_list = []
    for i in tqdm(range(len(dataset))):
        image, label = dataset[i]

        y = module(image)

        results = metrics.calculate_results(label, y, 5)
        results_list.append(results)

    pack = []
    for c in range(5):
        pack.append({
            'mask': 0,
            'pred': 0,
            'inter': 0,
            'union': 0,
        })

    for results in results_list:
        for c in range(5):
            pack[c]['mask'] += results[c]['mask']
            pack[c]['pred'] += results[c]['pred']
            pack[c]['inter'] += results[c]['inter']
            pack[c]['union'] += results[c]['union']

    print('%8s %12s %12s %12s %12s %8s %8s %8s' %
          ('class', 'mask', 'pred', 'inter', 'union', 'acc', 'iou', 'dice'))
    acc_list = []
    iou_list = []
    dice_list = []
    for c, r in enumerate(pack):
        acc = r['inter'] / r['mask']
        iou = r['inter'] / r['union']
        dice = 2 * r['inter'] / (r['mask'] + r['pred'])
        acc_list.append(acc)
        iou_list.append(iou)
        dice_list.append(dice)
        print('%8d %12d %12d %12d %12d %8.6f %8.6f %8.6f' %
              (c, r['mask'], r['pred'], r['inter'], r['union'], acc, iou, dice))

    accuracy = np.mean(acc_list)
    miou = np.mean(iou_list)
    dice_coeff = np.mean(dice_list)

    print('acc  ->', accuracy)
    print('miou ->', miou)
    print('dice ->', dice_coeff)


def evaluate2(
    data_root='./data/comma10k',
    model_path='./logs/comma10k/test1/models/model_050.pth',
    size=(640, 480),
    batch_size=8,
    num_workers=2,
):
    device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = 'mobilenet_v2'  # 'efficientnet-b0'
    net = smp.Unet(backbone, classes=5)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net = net.eval()

    T_val = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    val_dataset = datasets.Comma10k(data_root, False, T_val)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    pack = []
    for c in range(5):
        pack.append({
            'mask': 0,
            'pred': 0,
            'inter': 0,
            'union': 0,
        })

    criterion = nn.CrossEntropyLoss()

    losses = 0
    with torch.inference_mode():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)

            out = net(x)
            loss = criterion(out, y)

            losses += loss.detach()

            mask = y
            pred = torch.argmax(out, 1)

            results = []
            true = mask == pred
            for c in range(5):
                mask_c = mask == c
                pred_c = pred == c
                true_mask_c = true[mask_c]
                true_pred_c = true[pred_c]

                area_mask = len(true_mask_c)
                area_pred = len(true_pred_c)
                inter = torch.sum(true_mask_c)
                union = area_mask + area_pred - inter

                pack[c]['mask'] += area_mask
                pack[c]['pred'] += area_pred
                pack[c]['inter'] += inter
                pack[c]['union'] += union

    losses_val = losses / len(val_loader)
    print(losses_val)

    print('%8s %12s %12s %12s %12s %8s %8s %8s' %
          ('class', 'mask', 'pred', 'inter', 'union', 'acc', 'iou', 'dice'))
    acc_list = []
    iou_list = []
    dice_list = []
    for c, r in enumerate(pack):
        acc = r['inter'] / r['mask']
        iou = r['inter'] / r['union']
        dice = 2 * r['inter'] / (r['mask'] + r['pred'])
        acc_list.append(acc.cpu().numpy())
        iou_list.append(iou.cpu().numpy())
        dice_list.append(dice.cpu().numpy())
        print('%8d %12d %12d %12d %12d %8.6f %8.6f %8.6f' %
              (c, r['mask'], r['pred'], r['inter'], r['union'], acc, iou, dice))

    accuracy = np.mean(acc_list)
    miou = np.mean(iou_list)
    dice_coeff = np.mean(dice_list)

    print('acc  ->', accuracy)
    print('miou ->', miou)
    print('dice ->', dice_coeff)


if __name__ == '__main__':
    # evaluate()
    evaluate2(
        model_path='./logs/comma10k/test2/models/model_050.pth',
        size=(640, 480)
    )
