import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class ResNet9(nn.Module):
    def __init__(self, K=64, in_channels=1, base_width=32, normalize=False):
        super().__init__()
        self.normalize = normalize
        w = base_width

        self.conv1 = conv_bn_relu(in_channels, w,     kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_bn_relu(w,           w * 2, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.res1 = nn.Sequential(
            conv_bn_relu(w * 2, w * 2),
            conv_bn_relu(w * 2, w * 2)
        )

        self.conv3 = conv_bn_relu(w * 2, w * 4, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = conv_bn_relu(w * 4, w * 8, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.res2 = nn.Sequential(
            conv_bn_relu(w * 8, w * 8),
            conv_bn_relu(w * 8, w * 8)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(w * 8, K)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = x + self.res1(x)

        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.pool3(x)
        x = x + self.res2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x




