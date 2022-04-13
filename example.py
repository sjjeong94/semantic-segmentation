import json
from utils import *
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, deeplabv3_resnet101
import models


def train():
    set_seed(1234)

    learning_rate = 0.0003
    weight_decay = 0
    batch_size = 8
    epoch_begin = 0
    epoch_end = 100

    dataset = 'Cityscapes'
    name = 'test_002'
    dataset_root = f'./{dataset}/data'
    model_root = f'./{dataset}/model/{name}'
    result_root = f'./{dataset}/result'
    image_root = f'./{dataset}/image'

    #net = lraspp_mobilenet_v3_large(num_classes=20)
    net = models.MobileNetV2_FPN_FCN(256, 20)

    device = get_device()
    optimizer = torch.optim.Adam(
        net.to(device).parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    train_loader, val_loader = get_cityscapes(dataset_root, batch_size)

    print(net)
    print(device)
    print(optimizer)

    engine = Engine(
        name=name,
        net=net,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        model_root=model_root,
        result_root=result_root,
        image_root=image_root,
    )

    engine.train(epoch_begin, epoch_end, True)


def test():
    set_seed(1234)

    learning_rate = 0.0003
    weight_decay = 0
    batch_size = 16

    dataset = 'Cityscapes'
    name = 'test_001'
    dataset_root = f'./{dataset}/data'
    model_root = f'./{dataset}/model/{name}'
    result_root = f'./{dataset}/result'
    image_root = f'./{dataset}/image'

    net = lraspp_mobilenet_v3_large(num_classes=20)
    device = get_device()
    optimizer = torch.optim.Adam(
        net.to(device).parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    train_loader, val_loader = get_cityscapes(dataset_root, batch_size)

    net.load_state_dict(torch.load(
        'Cityscapes/model/test_001/test_001_epoch100.pth'))

    print(net)
    print(device)
    print(optimizer)

    engine = Engine(
        name=name,
        net=net,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        model_root=model_root,
        result_root=result_root,
        image_root=image_root,
    )

    engine.evaluate()


def train_comma10k():
    set_seed(1234)

    learning_rate = 0.0003
    weight_decay = 0
    batch_size = 32
    epoch_begin = 1
    epoch_end = 1

    dataset = 'Comma10k'
    name = 'test_002'
    model_root = f'./{dataset}/model/{name}'
    result_root = f'./{dataset}/result'
    image_root = f'./{dataset}/image'

    net = lraspp_mobilenet_v3_large(num_classes=5)
    #net = deeplabv3_resnet101(num_classes=5)

    device = get_device()
    optimizer = torch.optim.Adam(
        net.to(device).parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    train_loader, val_loader = get_comm10k(
        '../comma10k', batch_size, num_workers=4, pin_memory=True)

    print(net)
    print(device)
    print(optimizer)

    engine = Engine(
        name=name,
        net=net,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        model_root=model_root,
        result_root=result_root,
        image_root=image_root,
    )

    engine.train(epoch_begin, epoch_end, True)


def test_comma10k():
    set_seed(1234)

    learning_rate = 0.0003
    weight_decay = 0
    batch_size = 16

    dataset = 'Comma10k'
    name = 'test_001'
    model_root = f'./{dataset}/model/{name}'
    result_root = f'./{dataset}/result'
    image_root = f'./{dataset}/image'

    net = lraspp_mobilenet_v3_large(num_classes=5)
    device = get_device()
    optimizer = torch.optim.Adam(
        net.to(device).parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    train_loader, val_loader = get_comm10k('../comma10k', batch_size)

    net.load_state_dict(torch.load(
        'Comma10k/model/test_001/test_001_epoch000.pth'))

    print(net)
    print(device)
    print(optimizer)

    engine = Engine(
        name=name,
        net=net,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        model_root=model_root,
        result_root=result_root,
        image_root=image_root,
    )

    engine.evaluate()


def evaluate_comma10k():
    set_seed(1234)

    device = get_device()
    net = deeplabv3_resnet101(num_classes=5)
    net.load_state_dict(torch.load(
        'Comma10k/model/test_002/test_002_epoch001.pth'))
    net = net.to(device)
    net = net.eval()

    criterion = nn.CrossEntropyLoss()

    batch_size = 3
    T_val = T.Compose([
        T.RandomResize(512, 1024),
        T.RandomCrop(512),
        T.RandomHorizontalFlip(0.5)
    ])
    val_dataset = Comma10k('../comma10k', False, T_val)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)

    for _, (x, y) in enumerate(val_loader):
        images = x.numpy()
        lab = y.numpy()
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            out = net(x)['out']
            pred = torch.argmax(out, 1)
            pred = pred.cpu().numpy()

            loss = criterion(out, y)

        images = (images * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        images = np.concatenate(images, axis=1)

        pred_view = pred.astype(np.uint8) * 63
        pred_view = np.concatenate(pred_view, axis=1)

        error = pred != lab
        error_view = error.astype(np.uint8) * 255
        error_view = np.concatenate(error_view, axis=1)

        miou = get_miou(lab, pred, num_classes=5)
        print('Loss : %7.3f    mIoU : %7.3f' % (loss.item(), miou))

        cv2.imshow('images', images[:, :, ::-1])
        cv2.imshow('pred', pred_view)
        cv2.imshow('correct', error_view)

        key = cv2.waitKey(0)
        if key == 27:
            break


def demo():
    net = lraspp_mobilenet_v3_large(num_classes=20)
    net = models.MobileNetV2_FPN_FCN(256, 20)
    net.load_state_dict(torch.load(
        'Cityscapes/model/test_002/test_002_epoch100.pth'))

    cd = CityscapesDemo('../data/Cityscapes', net)
    cd.make_video('output_002.mp4')


if __name__ == '__main__':
    # train()
    # test()
    # demo()
    train_comma10k()
    # test_comma10k()
    # evaluate_comma10k()
