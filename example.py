from utils import *


def main():
    set_seed(1234)

    learning_rate = 0.0003
    weight_decay = 0
    batch_size = 16
    epoch_begin = 0
    epoch_end = 5

    dataset = 'Cityscapes'
    name = 'test'
    dataset_root = f'./{dataset}/data'
    model_root = f'./{dataset}/model/{name}'
    result_root = f'./{dataset}/result'
    image_root = f'./{dataset}/image'

    net = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        num_classes=34)
    device = get_device()
    optimizer = torch.optim.Adam(
        net.to(device).parameters(), lr=learning_rate, weight_decay=weight_decay)

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


if __name__ == '__main__':
    main()
