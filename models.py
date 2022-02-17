import torch
import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k_size,
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class DWConvBN(nn.Module):
    def __init__(self, in_ch, depth_multiplier, k_size, stride=1, padding=0):
        super().__init__()
        out_ch = in_ch * depth_multiplier
        self.conv = nn.Conv2d(in_ch, out_ch, k_size, stride,
                              padding, bias=False, groups=in_ch)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))

# TODO: add SE Block


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, exp_factor, s):
        super().__init__()
        self.act = nn.ReLU6()
        t_ch = int(in_ch * exp_factor)
        c_ch = int(out_ch)
        self.res = s == 1 and in_ch == out_ch
        if exp_factor == 1:
            self.do_conv1 = False
        else:
            self.do_conv1 = True
            self.conv1 = ConvBN(in_ch, t_ch, 1, 1)
        self.dwconv = DWConvBN(t_ch, 1, k_size, s, k_size // 2)
        self.conv2 = ConvBN(t_ch, c_ch, 1, 1)

    def forward(self, x):
        inputs = x
        if self.do_conv1:
            x = self.act(self.conv1(x))
        x = self.act(self.dwconv(x))
        x = self.act(self.conv2(x))
        if self.res:
            x = x + inputs
        return x


class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU6()
        self.conv1 = ConvBN(3, 32, 3, 2, 1)
        self.mb01 = MBConv(32, 16, 3, 1, 1)
        self.mb02 = MBConv(16, 24, 3, 6, 2)
        self.mb03 = MBConv(24, 24, 3, 6, 1)
        self.mb04 = MBConv(24, 32, 3, 6, 2)
        self.mb05 = MBConv(32, 32, 3, 6, 1)
        self.mb06 = MBConv(32, 32, 3, 6, 1)
        self.mb07 = MBConv(32, 64, 3, 6, 2)
        self.mb08 = MBConv(64, 64, 3, 6, 1)
        self.mb09 = MBConv(64, 64, 3, 6, 1)
        self.mb10 = MBConv(64, 64, 3, 6, 1)
        self.mb11 = MBConv(64, 96, 3, 6, 1)
        self.mb12 = MBConv(96, 96, 3, 6, 1)
        self.mb13 = MBConv(96, 96, 3, 6, 1)
        self.mb14 = MBConv(96, 160, 3, 6, 2)
        self.mb15 = MBConv(160, 160, 3, 6, 1)
        self.mb16 = MBConv(160, 160, 3, 6, 1)
        self.mb17 = MBConv(160, 320, 3, 6, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.mb01(x)
        x = self.mb02(x)
        x = self.mb03(x)
        o2 = x
        x = self.mb04(x)
        x = self.mb05(x)
        x = self.mb06(x)
        o3 = x
        x = self.mb07(x)
        x = self.mb08(x)
        x = self.mb09(x)
        x = self.mb10(x)
        x = self.mb11(x)
        x = self.mb12(x)
        x = self.mb13(x)
        o4 = x
        x = self.mb14(x)
        x = self.mb15(x)
        x = self.mb16(x)
        x = self.mb17(x)
        o5 = x
        return o2, o3, o4, o5


class FPN(nn.Module):
    def __init__(self, in_ch2, in_ch3, in_ch4, in_ch5, out_ch):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.act = nn.ReLU6()
        self.inner2 = ConvBN(in_ch2, out_ch, 1, 1)
        self.inner3 = ConvBN(in_ch3, out_ch, 1, 1)
        self.inner4 = ConvBN(in_ch4, out_ch, 1, 1)
        self.inner5 = ConvBN(in_ch5, out_ch, 1, 1)
        self.layer2 = ConvBN(out_ch, out_ch, 3, 1, 1)
        self.layer3 = ConvBN(out_ch, out_ch, 3, 1, 1)
        self.layer4 = ConvBN(out_ch, out_ch, 3, 1, 1)

    def forward(self, i2, i3, i4, i5):
        i2 = self.act(self.inner2(i2))
        i3 = self.act(self.inner3(i3))
        i4 = self.act(self.inner4(i4))
        o5 = self.act(self.inner5(i5))
        o4 = self.act(self.layer4(i4 + self.up(o5)))
        o3 = self.act(self.layer3(i3 + self.up(o4)))
        o2 = self.act(self.layer2(i2 + self.up(o3)))
        return o2, o3, o4, o5


class FCN(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch=1):
        super().__init__()
        self.act = nn.ReLU6()
        self.conv1 = ConvBN(in_ch, mid_ch, 3, 1, 1)
        self.drop = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x


class MobileNetV2_FPN(nn.Module):
    def __init__(self, out_ch=256):
        super().__init__()
        self.backbone = MobileNetV2()
        self.neck = FPN(24, 32, 96, 320, out_ch)

    def forward(self, x):
        o2, o3, o4, o5 = self.backbone(x)
        x = self.neck(o2, o3, o4, o5)
        return x


class MobileNetV2_FPN_FCN(nn.Module):
    def __init__(self, out_ch=256, num_cls=1):
        super().__init__()
        self.backbone_neck = MobileNetV2_FPN(out_ch)
        self.head = FCN(out_ch, out_ch, num_cls)
        self.up = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        o2, o3, o4, o5 = self.backbone_neck(x)
        x = self.head(o2)
        x = self.up(x)
        return {'out': x}


if __name__ == '__main__':
    #model = MobileNetV2_FPN(64)
    #sample = torch.rand((16, 3, 640, 480))
    #output = model(sample)
    # for out in output:
    #    print(out.shape)
    model = MobileNetV2_FPN_FCN(64, 5)
    sample = torch.rand((16, 3, 640, 480))
    output = model(sample)
    print(output['out'].shape)
    #torch.save(model.state_dict(), "model.pth")
