'''PredNet in PyTorch.'''

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PcConvBp(nn.Module):
    def __init__(self, inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.FFconv = nn.Conv2d(inchan, outchan, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        #self.cls = cls # e.g.: 5
        self.bypass = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        y = self.relu(self.FFconv(x))
        y = y + self.bypass(x)
        return y


class ClassifierModule(nn.Module):
    def __init__(self, in_channel_block, in_channel_clf, num_classes):
        super(ClassifierModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_channel_block)
        self.linear = nn.Linear(in_channel_block + in_channel_clf, num_classes)
        self.BN1d = nn.BatchNorm1d(num_classes)
    

    def forward(self, x_block):
        out_block = F.avg_pool2d(self.relu(self.BN(x_block)), x_block.size(-1))
        out_block = out_block.view(out_block.size(0), -1) # (batch_size, c_block)
        out = out_block

        rep = self.linear(out) # representation

        rep = self.BN1d(rep)

        return rep



''' Architecture PredNetBpD '''
class PredNetBpD(nn.Module):
    def __init__(self, num_classes=10):
        '''
            adaptive(bool): 
                True: Training with feedback, Testing without feedback
                False: Training with feedback, Testing with feedback
            vanilla(bool):
                True: No inputs from the previous classifiers
                False: Inputs from the previous classifiers
            ge(bool): Switch of Gradient Equilibrium
            fb (str):
                '1:1:1': open feedback for 3 classifiers
                '0:1:1': close feedback for 1st classifier
        '''
        super().__init__()
        self.ics = [3,  64, 64, 128, 128, 256, 256, 512] # input chanels
        self.ocs = [64, 64, 128, 128, 256, 256, 512, 512] # output chanels
        self.maxpool = [False, False, True, False, True, False, False, False] # downsample flag
        self.nlays = len(self.ics)
        self.classifiers = nn.ModuleList()

        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        #self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
        self.PcConvs = nn.ModuleList()
        for i in range(self.nlays):
            self.PcConvs.append(PcConvBp(self.ics[i], self.ocs[i]))
                
        self.classifier = ClassifierModule(in_channel_block=self.ocs[-1], in_channel_clf=0, num_classes=num_classes)
                
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.nlays)])
        # Linear layer
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        res = []
        for i in range(self.nlays):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x)  # ReLU + Conv
            if self.maxpool[i]:
                x = self.maxpool2d(x)

        res = self.classifier(x)

        return res

