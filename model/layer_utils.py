from math import ceil
import torch.nn.functional as F
import torch
from torch import nn


def conv_bn_relu(input_channel, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.PReLU()
    )


def conv_bn_no_relu(input_channel, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output_channel)
    )


def conv_bn_relu_1x1(input_channel, output_channel):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.PReLU()
    )


def conv_dw(input_channel, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(input_channel, input_channel, 3, stride, 1, groups=input_channel, bias=False),
        nn.BatchNorm2d(input_channel),
        nn.PReLU(),

        nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.PReLU()
    )


class MobileNetV1(nn.Module):
    def __init__(self, size):
        super(MobileNetV1, self).__init__()
        self.size = size
        self.stage1 = nn.Sequential(
            conv_bn_relu(3, ceil(32 * size), 2),
            conv_dw(ceil(32 * size), ceil(64 * size), 1),
            conv_dw(ceil(64 * size), ceil(128 * size), 2),
            conv_dw(ceil(128 * size), ceil(128 * size), 1),
            conv_dw(ceil(128 * size), ceil(256 * size), 2),
            conv_dw(ceil(256 * size), ceil(256 * size), 1),
        )
        self.stage2 = nn.Sequential(
            conv_dw(ceil(256 * size), ceil(512 * size), 2),
            conv_dw(ceil(512 * size), ceil(512 * size), 1),
            conv_dw(ceil(512 * size), ceil(512 * size), 1),
            conv_dw(ceil(512 * size), ceil(512 * size), 1),
            conv_dw(ceil(512 * size), ceil(512 * size), 1),
            conv_dw(ceil(512 * size), ceil(512 * size), 1)
        )
        self.stage3 = nn.Sequential(
            conv_dw(ceil(512 * size), ceil(1024 * size), 2),
            conv_dw(ceil(1024 * size), ceil(1024 * size), 1)
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ceil(1024 * size), 1000)

    def forward(self, x):
        stage1_x = self.stage1(x)
        stage2_x = self.stage2(stage1_x)
        stage3_x = self.stage3(stage2_x)
        out = self.avg(stage3_x)
        out = out.reshape(-1, out.shape[1])
        out = self.fc(out)
        return out


class FPN(nn.Module):
    def __init__(self, input_channel_list, output_channels):
        super(FPN, self).__init__()

        self.output1 = conv_bn_relu_1x1(input_channel_list[0], output_channels)
        self.output2 = conv_bn_relu_1x1(input_channel_list[1], output_channels)
        self.output3 = conv_bn_relu_1x1(input_channel_list[2], output_channels)

        self.merge1 = conv_bn_relu(output_channels, output_channels, 1)
        self.merge2 = conv_bn_relu(output_channels, output_channels, 1)

    def forward(self, input):
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class SSH(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SSH, self).__init__()

        self.conv3x3 = conv_bn_no_relu(input_channels, output_channels // 2, 1)

        self.conv5x5_1 = conv_bn_relu(input_channels, output_channels // 4, 1)
        self.conv5x5_2 = conv_bn_no_relu(output_channels // 4, output_channels // 4, 1)

        self.conv7x7_2 = conv_bn_relu(output_channels // 4, output_channels // 4, 1)
        self.conv7x7_3 = conv_bn_no_relu(output_channels // 4, output_channels // 4, 1)

    def forward(self, input):
        conv3x3 = self.conv3x3(input)
        conv5x5_1 = self.conv5x5_1(input)
        conv5x5 = self.conv5x5_2(conv5x5_1)
        conv7x7_2 = self.conv7x7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = torch.relu(out)
        return out


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        return out.reshape(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(BboxHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        return out.reshape(out.shape[0], -1, 4)


class LandmarksHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(LandmarksHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        return out.reshape(out.shape[0], -1, 10)


if __name__ == '__main__':
    net = MobileNetV1(0.25)
    x = torch.randn(1, 3, 640, 640)
    print(net(x).shape)
