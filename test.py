import cv2
import videocv
import numpy as np
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_path = './logs/comma10k/test/models/model_050.pth'
    net = deeplabv3_mobilenet_v3_large(num_classes=5)
    net.load_state_dict(torch.load(net_path))
    net = net.eval()
    net = net.to(device)

    w, h = 640, 320
    video = videocv.Video('test.mp4')
    while video():
        frame = video.frame
        image = frame[-1920:]
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32).transpose(2, 0, 1) / 255.
        x = x[None]
        x = torch.from_numpy(x)
        x = x.to(device)
        with torch.inference_mode():
            out = net(x)['out']
            out = torch.argmax(out, dim=1).squeeze().cpu().numpy()

        print(out.shape)
        view = (out * 60).astype(np.uint8)
        cv2.imshow('image', image)
        cv2.imshow('viwe', view)


if __name__ == '__main__':
    main()
