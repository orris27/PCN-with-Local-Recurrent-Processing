'''PredNet in PyTorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PcConvBp(nn.Module):
    def __init__(self, inchan, outchan, kernel_size=3, stride=1, padding=1, cls=0, bias=False):
        super().__init__()
        self.FFconv = nn.Conv2d(inchan, outchan, kernel_size, stride, padding, bias=bias)
        self.FBconv = nn.ConvTranspose2d(outchan, inchan, kernel_size, stride, padding, bias=bias)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,outchan,1,1))])
        self.relu = nn.ReLU(inplace=True)
        self.cls = cls # e.g.: 5
        self.bypass = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        y = self.relu(self.FFconv(x))
        b0 = F.relu(self.b0[0]+1.0).expand_as(y)
        for _ in range(self.cls):
            y = self.FFconv(self.relu(x - self.FBconv(y)))*b0 + y
        y = y + self.bypass(x)
        return y


class ClassifierModule(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_channel)
        self.linear = nn.Linear(in_channel, num_classes)

    def forward(self, x): # x (list)
        out = F.avg_pool2d(self.relu(self.BN(x)), x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


''' Architecture PredNetBpD '''
class PredNetBpD(nn.Module):
    def __init__(self, num_classes=10, cls=0, Tied = False):
        super().__init__()
        self.ics = [3,  64, 64, 128, 128, 256, 256, 512] # input chanels
        self.ocs = [64, 64, 128, 128, 256, 256, 512, 512] # output chanels
        self.maxpool = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)
        self.classifiers = nn.ModuleList()

        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        if Tied == False:
            #self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
            print('Tied: False')
            self.PcConvs = nn.ModuleList()
            for i in range(self.nlays):
                self.PcConvs.append(PcConvBp(self.ics[i], self.ocs[i], cls=self.cls))
                self.classifiers.append(ClassifierModule(self.ocs[i], num_classes))
                
                
        else:
            print('Tied: True')
            self.PcConvs = nn.ModuleList([PcConvBpTied(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.nlays)])
        # Linear layer
        #self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.relu = nn.ReLU(inplace=True)
        #self.BNend = nn.BatchNorm2d(self.ocs[-1])




    def forward(self, x):
        res = []
        for i in range(self.nlays):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x)  # ReLU + Conv
            if self.maxpool[i]:
                x = self.maxpool2d(x)

            res.append(self.classifiers[i](x))

        # classifier                
        #out = F.avg_pool2d(self.relu(self.BNend(x)), x.size(-1))
        #out = out.view(out.size(0), -1)
        #out = self.linear(out)
        #return out
        return res

