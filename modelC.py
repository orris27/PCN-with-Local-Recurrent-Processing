'''PredNet in PyTorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PcConvBp(nn.Module):
    def __init__(self, inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.FFconv = nn.Conv2d(inchan, outchan, kernel_size, stride, padding, bias=bias)
        self.FBconv = nn.ConvTranspose2d(outchan, inchan, kernel_size, stride, padding, bias=bias)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,outchan,1,1))])
        self.relu = nn.ReLU(inplace=True)
        #self.cls = cls # e.g.: 5
        self.bypass = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        y = self.relu(self.FFconv(x))
        b0 = F.relu(self.b0[0]+1.0).expand_as(y)
        #for _ in range(self.cls):
        #    y = self.FFconv(self.relu(x - self.FBconv(y)))*b0 + y
        y = y + self.bypass(x)
        return y


class ClassifierModule(nn.Module):
    def __init__(self, in_channel_block, in_channel_clf, num_classes, adaptive, cls, dropout):
        super(ClassifierModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_channel_block)
        self.linear = nn.Linear(in_channel_block + in_channel_clf, num_classes)
        self.linear_bw = nn.Linear(num_classes, in_channel_block + in_channel_clf)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,num_classes))])
        self.cls = cls # e.g.: 5
        self.adaptive = adaptive
        self.dropout = dropout
    

    def forward(self, x_block, x_clf):
        '''
            x_block (4D Tensor): from block
            x_clf (2D Tensor): from previous classifier

        '''
        out_block = F.avg_pool2d(self.relu(self.BN(x_block)), x_block.size(-1))
        out_block = out_block.view(out_block.size(0), -1) # (batch_size, c_block)
        
        out_clf = x_clf # (batch_size, c_clf)

        if x_clf is None:
            out = out_block
        else:
            out = torch.cat([out_block, out_clf], dim=1)

        rep = self.linear(out) # representation

        b0 = F.relu(self.b0[0] + 1.0).expand_as(rep)
        if self.adaptive is True and self.training is False:
            #print('No feedback')
            pass
        else:
            if torch.distributions.Bernoulli(torch.tensor(self.dropout)).sample() == 1:
                for _ in range(self.cls):
                    rep = self.linear(self.relu(out - self.linear_bw(rep))) * b0 + rep

        # no bypass

        return rep


''' Architecture PredNetBpD '''
class PredNetBpD(nn.Module):
    def __init__(self, num_classes=10, cls=0, dropout=1.0, adaptive = False, avg=False):
        '''
            adaptive(bool): 
                True: Training with feedback, Testing without feedback
                False: Training with feedback, Testing with feedback
            avg(bool):
                True: feed the outputs from clf_i to clf_{i+1}
                False: feed the average outputs from clf_0, clf_1, ..., clf_i to clf_{i+1}
        '''
        super().__init__()
        self.ics = [3,  64, 64, 128, 128, 256, 256, 512] # input chanels
        self.ocs = [64, 64, 128, 128, 256, 256, 512, 512] # output chanels
        self.maxpool = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)
        self.adaptive = adaptive # True: training adopts feedback, but testing not; False: both training and testing uses feedbacks
        self.classifiers = nn.ModuleList()
        self.dropout = dropout
        self.avg = avg

        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        #self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
        self.PcConvs = nn.ModuleList()
        for i in range(self.nlays):
            self.PcConvs.append(PcConvBp(self.ics[i], self.ocs[i]))
            if self.maxpool[i] is True:
                if len(self.classifiers) == 0:
                    self.classifiers.append(ClassifierModule(in_channel_block=self.ocs[i], in_channel_clf=0, num_classes=num_classes, adaptive=self.adaptive, cls=self.cls, dropout=self.dropout))
                else:
                    self.classifiers.append(ClassifierModule(in_channel_block=self.ocs[i], in_channel_clf=num_classes, num_classes=num_classes, adaptive=self.adaptive, cls=self.cls, dropout=self.dropout))
                
        self.classifiers.append(ClassifierModule(in_channel_block=self.ocs[-1], in_channel_clf=num_classes, num_classes=num_classes, adaptive=self.adaptive, cls=self.cls, dropout=self.dropout))
                
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

                # add classifier
                if len(res) == 0:
                    res.append(self.classifiers[len(res)](x, None))
                else:
                    if self.avg is True:
                        res.append(self.classifiers[len(res)](x, sum(res) / len(res)))
                    else:
                        res.append(self.classifiers[len(res)](x, res[-1]))

                x = self.maxpool2d(x)

        if self.avg is True:
            res.append(self.classifiers[len(res)](x, sum(res) / len(res)))
        else:
            res.append(self.classifiers[len(res)](x, res[-1]))

        # classifier                
        #out = F.avg_pool2d(self.relu(self.BNend(x)), x.size(-1))
        #out = out.view(out.size(0), -1)
        #out = self.linear(out)
        #return out
        return res

