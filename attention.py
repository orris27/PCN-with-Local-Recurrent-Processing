import torch
from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class ScanLayer(nn.Module):
    def __init__(self, channel):
        super(ScanLayer, self).__init__()
        self.conv = nn.Conv2d(channel, channel, 2, 2)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

        self.deconv = nn.ConvTranspose2d(channel, channel, 2, 2)
        self.bn_2 = nn.BatchNorm2d(channel)


    def forward(self, x):
        y = self.conv(x)
        y = self.relu(self.bn_1(y))

        y = self.deconv(y)
        y = torch.sigmoid(self.bn_2(y))

        return x * y

class LinearLayer(nn.Module):
    def __init__(self, channel):
        super(LinearLayer, self).__init__()
        self.attention = attention = nn.Sequential(
                    nn.Linear(channel, channel // 4), 
                    nn.BatchNorm1d(channel // 4), 
                    nn.ReLU(), 
                    nn.Linear(channel // 4, channel), 
                    nn.BatchNorm1d(channel), 
                    nn.Sigmoid())                                                                                          



    def forward(self, x):
        y = self.attention(x)
        return x * y



if __name__ == '__main__':
    # ScanLayer
    x = torch.randn(2, 64, 112, 112)
    scan = ScanLayer(64)
    print(scan(x).shape) # (2, 64, 112, 112)


    x = torch.randn(2, 16)
    linear = LinearLayer(16)
    print(linear(x).shape) # (2, 64, 112, 112)
