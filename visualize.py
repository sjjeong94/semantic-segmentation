import cv2
import numpy as np
import videocv
import torch
import torchvision
import segmentation_models_pytorch as smp

import datasets


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
        with torch.inference_mode():
            out = self.net(x)
            y = torch.argmax(out, 1)
            y = y.cpu().numpy().squeeze()
        return y


def main(
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


if __name__ == '__main__':
    main()
