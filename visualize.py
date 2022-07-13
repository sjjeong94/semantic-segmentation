import cv2
import numpy as np
import videocv
import torch
import torchvision
import segmentation_models_pytorch as smp

import datasets
import transforms
import metrics


class Module:
    def __init__(self, model_path):

        device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        backbone = 'mobilenet_v2'  # 'efficientnet-b0'
        net = smp.Unet(backbone, classes=5)

        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
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


def visualze_video(
    video_path='./videos/test.mp4',
    save_path='./videos/segmentation.mp4',
    model_path='./logs/comma10k/test/models/model_050.pth',
    size=(1024, 576),
):

    module = Module(model_path)

    video = videocv.Video(video_path)
    writer = videocv.Writer(save_path, video.fps, (size[0]*2, size[1]*2))

    idx = 0
    while video():
        idx += 1
        print('%d / %d' % (idx, video.frame_count))

        image = cv2.resize(video.frame, size, interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        y = module(x)

        mask = datasets.label_decode(y)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

        edge = cv2.Laplacian(y.astype(np.uint8), -1)
        edge = edge != 0

        overlap = image // 2 + mask // 2
        overlap[edge] = 255

        boundary = np.zeros(image.shape, np.uint8)
        boundary[edge] = (0, 255, 255)

        view_top = np.concatenate([image, mask], axis=1)
        view_bot = np.concatenate([overlap, boundary], axis=1)
        view = np.concatenate([view_top, view_bot])
        writer(view)

        cv2.imshow('view', view)


def visualize_eval(
    data_root='./data/comma10k',
    model_path='./logs/comma10k/test2/models/model_050.pth',
    size=(640, 480),
):

    module = Module(model_path)

    dataset = datasets.Comma10k(
        data_root,
        False,
        transforms.Resize(size)
    )

    idx = 0
    while True:
        image, label = dataset[idx]

        y = module(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(datasets.label_decode(label), cv2.COLOR_RGB2BGR)
        pred = cv2.cvtColor(datasets.label_decode(y), cv2.COLOR_RGB2BGR)

        results = metrics.calculate_results(label, y, 5)
        print('%8s %8s %8s %8s %8s' %
              ('class', 'mask', 'pred', 'inter', 'union'))
        for c, r in enumerate(results):
            print('%8d %8d %8d %8d %8d' %
                  (c, r['mask'], r['pred'], r['inter'], r['union']))
        print()

        #error = np.zeros(image.shape, np.uint8)
        #error[y != label] = (0, 255, 255)

        error = mask.copy()
        error[y == label] = 0

        view_top = np.concatenate([image, error], axis=1)
        view_bot = np.concatenate([mask, pred], axis=1)
        view = np.concatenate([view_top, view_bot])

        cv2.imshow('view', view)

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


if __name__ == '__main__':
    visualze_video(
        model_path='./logs/comma10k/releases/v220622_1.pt',
    )
    # visualize_eval()
