import torch.nn as nn
import torch.nn.functional as F

import config
from utils import error_callback


class CAN(nn.Module):

    def __init__(self, no_of_filters=config.can_filter_count):
        super(CAN, self).__init__()

        # CAN24 architecture: 3 RGB + 8 Filter Channels -> 11 InChannels
        in_count = no_of_filters + 3
        self.conv1 = nn.Conv2d(in_channels=in_count, out_channels=24, kernel_size=(3, 3), dilation=1,
                               padding=(1, 1))  # weight shape [24,8,3,3]
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), dilation=2,
                               padding=(2, 2))  # weight shape [24,24,3,3]
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), dilation=4,
                               padding=(4, 4))  # weight shape     ""
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), dilation=8,
                               padding=(8, 8))  # weight shape     ""
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), dilation=16,
                               padding=(16, 16))  # weight shape     ""
        self.conv6 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), dilation=32,
                               padding=(32, 32))  # weight shape     ""
        self.conv7 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), dilation=64,
                               padding=(64, 64))  # weight shape     ""
        self.conv9 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), dilation=1,
                               padding=(1, 1))  # weight shape     ""
        self.conv10 = nn.Conv2d(in_channels=24, out_channels=3, kernel_size=(1, 1),
                                dilation=1)  # weight shape [3,24,1,1]

    def forward(self, x):
        inshape = x.shape
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv6(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv7(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv9(x), negative_slope=0.2)
        x = self.conv10(x)  # no activation in last layer

        if inshape[-2] != x.shape[-2] or inshape[-1] != x.shape[-1]:
            error_callback('forward_conv')

        return x


class NIMA_VGG(nn.Module):
    def __init__(self, base_model, num_classes=10):
        super(NIMA_VGG, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(  # self.classifier describes only the last layer
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
