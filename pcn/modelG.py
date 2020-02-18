'''PredNet in PyTorch.'''

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---- GradientRescale ---- #
class GradientRescaleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        ctx.gd_scale_weight = weight
        output = input
        return output


    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = ctx.gd_scale_weight * grad_output

        return grad_input, grad_weight

gradient_rescale = GradientRescaleFunction.apply

# ---- END Gradient Rescale ---- #


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
    def __init__(self, in_channel_block, in_channel_clf, num_classes, cls, hidden_channel=0):
        super(ClassifierModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_channel_block)

        if hidden_channel != 0:
            self.h = True # contains hidden layer
            self.linear_h = nn.Linear(in_channel_block + in_channel_clf, hidden_channel)
            self.linear = nn.Linear(hidden_channel, num_classes)
        else:
            self.h = False
            self.linear = nn.Linear(in_channel_block + in_channel_clf, num_classes)


        self.cls = cls
        if self.cls != 0:
            self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,num_classes))])
            if hidden_channel != 0:
                self.linear_bw = nn.Linear(num_classes, hidden_channel)
            else:
                self.linear_bw = nn.Linear(num_classes, in_channel_block + in_channel_clf)
        self.BN1d = nn.BatchNorm1d(num_classes)
    

    def forward(self, x_block, x_clf):
        out_block = F.avg_pool2d(self.relu(self.BN(x_block)), x_block.size(-1))
        out_block = out_block.view(out_block.size(0), -1) # (batch_size, c_block)
        out_clf = x_clf # (batch_size, c_clf)

        if x_clf is None:
            out = out_block
        else:
            out = torch.cat([out_block, out_clf], dim=1)

        if self.h is True:
            out = self.linear_h(out)

        rep = self.linear(out) # representation

        if self.cls != 0: # feedback
            b0 = F.relu(self.b0[0] + 1.0).expand_as(rep)
            for _ in range(self.cls):
                rep = self.linear(self.relu(out - self.linear_bw(rep))) * b0 + rep

        rep = self.BN1d(rep)

        return rep


''' Architecture PredNetBpD '''
class PredNetBpD(nn.Module):
    def __init__(self, num_classes=10, cls=0, ge=False):
        '''
            ge(bool): Switch of Gradient Equilibrium
        '''
        super().__init__()
        self.ics = [3,  64, 64, 128, 128, 256, 256, 512] # input chanels
        self.ocs = [64, 64, 128, 128, 256, 256, 512, 512] # output chanels
        self.maxpool = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)
        self.classifiers = nn.ModuleList()
        self.ge = ge

        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        #self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
        self.PcConvs = nn.ModuleList()
        for i in range(self.nlays):
            self.PcConvs.append(PcConvBp(self.ics[i], self.ocs[i]))
                
        self.classifiers.append(ClassifierModule(in_channel_block=128, in_channel_clf=0, num_classes=num_classes, cls=0, hidden_channel=32))
        self.classifiers.append(ClassifierModule(in_channel_block=256, in_channel_clf=num_classes, num_classes=num_classes, cls=self.cls, hidden_channel=72))
        self.classifiers.append(ClassifierModule(in_channel_block=512, in_channel_clf=0, num_classes=num_classes, cls=0))

                
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.nlays)])
        # Linear layer
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)



    def forward(self, x):
        res = []
        for i in range(self.nlays):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x)  # ReLU + Conv
            if self.maxpool[i] or i == self.nlays - 1:

                clf_id = len(res)
                # add classifier

                if self.ge is True:
                    x = gradient_rescale(x, 1.0 / (len(self.classifiers) - clf_id))

                if len(res) == 0 or len(res) == 2:
                    res.append(self.classifiers[clf_id](x, None))
                else:
                    res.append(self.classifiers[clf_id](x, res[-1]))

                if self.ge is True:
                    x = gradient_rescale(x, (len(self.classifiers) - clf_id - 1))

                if self.maxpool[i] is True:
                    x = self.maxpool2d(x)

        return res

