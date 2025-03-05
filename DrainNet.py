import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class DirectConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        kernel = 5
        self.conv_v = nn.Sequential(
            nn.Conv2d(inchannel, outchannel//2, (kernel, 1), padding=((kernel-1)//2, 0)),
            nn.BatchNorm2d(outchannel//2),
            nn.ReLU(inplace=True),
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(inchannel, outchannel//2, (1, kernel), padding=(0, (kernel-1)//2)),
            nn.BatchNorm2d(outchannel//2),
            nn.ReLU(inplace=True),
        )
        self.c_attention = ChannelAttention(outchannel)
    def forward(self, x):
        out1 = self.conv_v(x)
        out2 = self.conv_h(x)
        out = torch.cat([out1, out2], dim=1)
        out = out*self.c_attention(out)
        return out

class DilatedConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 5, 1, 4, dilation=2),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, 5, 1, 4, dilation=2),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class DrainNet(nn.Module):
    def __init__(self, inchannel, out_channels, channels=[16, 32, 64, 128, 256]):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(inchannel, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv = nn.Sequential(
            DirectConv(channels[0], channels[0]),
            DilatedConv(channels[0], channels[0]),
            DirectConv(channels[0], channels[1]),
            DilatedConv(channels[1], channels[1]),
            DirectConv(channels[1], channels[2]),
            DilatedConv(channels[2], channels[2]),
            # DirectConv(channels[2], channels[3]),
            # DilatedConv(channels[3], channels[3]),
            # DirectConv(channels[3], channels[4]),
            # DilatedConv(channels[4], channels[4]),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.tail = nn.Conv2d(channels[-1], out_channels, 1)
    def forward(self, x):
        x = self.head(x)
        x = self.conv(x)
        x = self.up(x)
        out = self.tail(x)
        return out
