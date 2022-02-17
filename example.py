from utils import *
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
import models


def main():
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
    epoch_begin = 0
    epoch_end = 100

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


def demo():
    #net = lraspp_mobilenet_v3_large(num_classes=20)
    net = models.MobileNetV2_FPN_FCN(256, 20)
    net.load_state_dict(torch.load(
        'Cityscapes/model/test_002/test_002_epoch100.pth'))
    cd = CityscapesDemo('Cityscapes/data', net)
    cd.make_video('output_002.mp4')


if __name__ == '__main__':
    # main()
    # test()
    demo()
