import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Residual(nn.Module):  # pytorch
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

class SSRN_network(nn.Module):
    def __init__(self, band, classes):
        super(SSRN_network, self).__init__()
        self.name = 'SSRN'
        self.conv1 = nn.Conv2d(in_channels=24, out_channels=24,
                               kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.res_net1 = Residual(24, 24, (7, 7), (3, 3))
        self.res_net2 = Residual(24, 24, (7, 7), (3, 3))
        self.res_net3 = Residual(24, 24, (3, 3), (1, 1))
        self.res_net4 = Residual(24, 24, (3, 3), (1, 1))

        # Adjust kernel size here based on input dimensions after convolution
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=128, padding=(0, 0),
                               kernel_size=(math.ceil((band - 6) / 2), math.ceil((band - 6) / 2)), stride=(1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Reduce kernel size for conv3 based on input dimensions
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=24, padding=(0, 0),
                               kernel_size=(1, 1), stride=(1, 1))  # Adjusted kernel size
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.full_connection = nn.Sequential(
            nn.Linear(24, classes)
        )

    def forward(self, X):
        x1 = self.batch_norm1(self.conv1(X))
        x2 = self.res_net3(x1)
        x2 = self.res_net4(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = self.batch_norm3(self.conv3(x2))

        x2 = self.global_avg_pooling(x2)  # Global average pooling to get (batch_size, 24, 1, 1)
        x2 = torch.flatten(x2, 1)  # Flatten to (batch_size, 24)

        x2 = self.full_connection(x2)  # Linear layer to get the final output (batch_size, classes)

        return x2









