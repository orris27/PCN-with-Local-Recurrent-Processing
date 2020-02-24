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
        self.relu = nn.ReLU(inplace=False)
        #self.cls = cls # e.g.: 5
        self.bypass = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        y = self.relu(self.FFconv(x))
        y = y + self.bypass(x)
        return y


class ClassifierModuleFirst(nn.Module):
    def __init__(self, in_channel_block, num_classes, hidden_channel, score_layer=False):
        super(ClassifierModuleFirst, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.BN = nn.BatchNorm2d(in_channel_block)

        self.linear_h = nn.Linear(in_channel_block, hidden_channel)
        self.linear = nn.Linear(hidden_channel, num_classes)

        self.BN1d = nn.BatchNorm1d(hidden_channel)

        self.score_layer = score_layer
        if self.score_layer is True:
            self.score = nn.Linear(self.linear.in_features, 1)

    def forward(self, x_block):
        out_block = F.avg_pool2d(self.relu(self.BN(x_block)), x_block.size(-1))
        out_block = out_block.view(out_block.size(0), -1) # (batch_size, c_block)

        out = out_block
        h = self.BN1d(self.linear_h(out))

        out = self.linear(self.relu(h))
        if self.score_layer is True:
            score = torch.sigmoid(self.score(self.relu(h)))
        else:
            score = None

        return out, h, score

class ClassifierModuleMiddle(nn.Module):
    def __init__(self, in_channel_block, in_channel_clf, num_classes, cls, hidden_channel, score_layer=False):
        super(ClassifierModuleMiddle, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.BN = nn.BatchNorm2d(in_channel_block)

        self.linear_h = nn.Linear(in_channel_block + in_channel_clf, hidden_channel)
        self.linear = nn.Linear(hidden_channel, num_classes)


        self.cls = cls
        if self.cls != 0:
            self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, hidden_channel))])
            self.linear_bw = nn.Linear(hidden_channel, in_channel_block + in_channel_clf)
        self.BN1d = nn.BatchNorm1d(hidden_channel)

        self.score_layer = score_layer
        if self.score_layer is True:
            self.score = nn.Linear(self.linear.in_features, 1)

    

    def forward(self, x_block, x_clf):
        out_block = F.avg_pool2d(self.relu(self.BN(x_block)), x_block.size(-1))
        out_block = out_block.view(out_block.size(0), -1) # (batch_size, c_block)
        out_clf = x_clf # (batch_size, c_clf)

        out = torch.cat([out_block, out_clf], dim=1)
            
        # out -> h
        rep = self.linear_h(out)
        if self.cls == 0:
            pass
        else:
            b0 = F.relu(self.b0[0] + 1.0).expand_as(rep)
            for _ in range(self.cls):
                rep = self.linear_h(self.relu(out - self.linear_bw(rep))) * b0 + rep
        h = self.BN1d(rep)


        out = self.linear(self.relu(h))
        if self.score_layer is True:
            score = torch.sigmoid(self.score(self.relu(h)))
        else:
            score = None

        return out, h, score

class ClassifierModuleLast(nn.Module):
    def __init__(self, in_channel_block, num_classes):
        super(ClassifierModuleLast, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.BN = nn.BatchNorm2d(in_channel_block)

        self.linear = nn.Linear(in_channel_block, num_classes)

    def forward(self, x_block):
        out_block = F.avg_pool2d(self.relu(self.BN(x_block)), x_block.size(-1))
        out_block = out_block.view(out_block.size(0), -1) # (batch_size, c_block)

        out = out_block
        out = self.linear(out)

        return out




''' Architecture PredNetBpD '''
class PredNetBpD(nn.Module):
    def __init__(self, num_classes=10, cls=0, ge=False, score_layer=False):
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
        self.score_layer = score_layer

        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        #self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
        self.PcConvs = nn.ModuleList()
        for i in range(self.nlays):
            self.PcConvs.append(PcConvBp(self.ics[i], self.ocs[i]))
                
        self.classifiers.append(ClassifierModuleFirst(in_channel_block=128, num_classes=num_classes, hidden_channel=32, score_layer=self.score_layer))
        self.classifiers.append(ClassifierModuleMiddle(in_channel_block=256, in_channel_clf=32, num_classes=num_classes, cls=self.cls, hidden_channel=72, score_layer=self.score_layer))
        self.classifiers.append(ClassifierModuleLast(in_channel_block=512, num_classes=num_classes))

                
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.nlays)])
        # Linear layer
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)



    def forward(self, x):
        res = []
        scores = []
        for i in range(self.nlays):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x)  # ReLU + Conv
            if self.maxpool[i] or i == self.nlays - 1:

                clf_id = len(res)
                # add classifier

                if self.ge is True:
                    x = gradient_rescale(x, 1.0 / (len(self.classifiers) - clf_id))

                if len(res) == 0: # first classifier
                    r, h, score = self.classifiers[clf_id](x)
                    res.append(r)
                    scores.append(score)
                elif len(res) < len(self.classifiers) - 1: # middle classifiers
                    r, h, score = self.classifiers[clf_id](x, h)
                    res.append(r)
                    scores.append(score)
                else: # last classifiers
                    r = self.classifiers[clf_id](x)
                    res.append(r)

                if self.ge is True:
                    x = gradient_rescale(x, (len(self.classifiers) - clf_id - 1))

                if self.maxpool[i] is True:
                    x = self.maxpool2d(x)

        return res, scores

