'''PredNet in PyTorch.'''

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
    def __init__(self, in_channel_block, in_channel_clf, hidden_channel, num_classes, adaptive, cls, dropout):
        super(ClassifierModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_channel_block)
        
        self.FFconv = nn.Conv2d(in_channel_block + in_channel_clf, hidden_channel, kernel_size=3, stride=1, padding=1, bias=False) 

        self.linear = nn.Linear(hidden_channel, num_classes)
        self.cls = cls # e.g.: 5
        if self.cls != 0:
            self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,hidden_channel, 1, 1))])
            self.FBconv = nn.ConvTranspose2d(hidden_channel, in_channel_block + in_channel_clf, kernel_size=3, stride=1, padding=1, bias=False) 
        self.adaptive = adaptive
        self.dropout = dropout
        self.bypass = nn.Conv2d(in_channel_block + in_channel_clf, hidden_channel, kernel_size=1, stride=1, bias=False)
    

    def forward(self, x_block, x_clf):
        '''
            x_block (4D Tensor): from block (2, 128, 32, 32),(2, 256, 16, 16), (2, 512, 8, 8)
            x_clf (4D Tensor): from previous classifier

        '''
        out_block = self.relu(self.BN(x_block)) # e.g. (batch_size, 256, 16, 16)
        
        # x_clf: e.g. (batch_size, 256, 16, 16)
        out_clf = x_clf

        if x_clf is None:
            out = out_block
        else:
            out = torch.cat([out_block, out_clf], dim=1)

        rep = self.relu(self.FFconv(out)) # (batch_size, 256 + 256, x, x)
        if self.cls == 0 or (self.adaptive is True and self.training is False):
            pass
        else:
            if torch.distributions.Bernoulli(torch.tensor(self.dropout)).sample() == 1:
                b0 = F.relu(self.b0[0] + 1.0).expand_as(rep)
                for _ in range(self.cls):
                    rep = self.FFconv(self.relu(out - self.FBconv(rep))) * b0 + rep

        rep = F.max_pool2d(rep + self.bypass(out), 2)
        return rep, self.linear(F.avg_pool2d(rep, rep.size(-1)).squeeze(-1).squeeze(-1))
       
        

''' Architecture PredNetBpD '''
class PredNetBpD(nn.Module):
    def __init__(self, num_classes=10, cls=0, dropout=1.0, adaptive=False, vanilla=False, ge=False, fb=None):
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
        self.clf_h = [128, 256, 512] # hidden channels for classifiers

        self.maxpool = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)
        self.adaptive = adaptive # True: training adopts feedback, but testing not; False: both training and testing uses feedbacks
        self.classifiers = nn.ModuleList()
        self.dropout = dropout
        self.vanilla = vanilla
        self.ge = ge

        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        self.PcConvs = nn.ModuleList()
        for i in range(self.nlays):
            self.PcConvs.append(PcConvBp(self.ics[i], self.ocs[i]))
            if self.maxpool[i] is True or i + 1 == self.nlays:
                if self.vanilla is True or len(self.classifiers) == 0:
                    #def __init__(self, in_channel_block, in_channel_clf, hidden_channel, num_classes, adaptive, cls, dropout):
                    self.classifiers.append(ClassifierModule(in_channel_block=self.ocs[i], in_channel_clf=0, hidden_channel=self.clf_h[len(self.classifiers)], num_classes=num_classes, adaptive=self.adaptive, cls=self.cls, dropout=self.dropout))
                else:
                    self.classifiers.append(ClassifierModule(in_channel_block=self.ocs[i], in_channel_clf=self.clf_h[len(self.classifiers) - 1], hidden_channel=self.clf_h[len(self.classifiers)], num_classes=num_classes, adaptive=self.adaptive, cls=self.cls, dropout=self.dropout))
                
                
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.nlays)])
        # Linear layer
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        if fb is not None:
            fb_switch = list(map(bool, map(int, fb.split(':'))))
            for idx, elm in enumerate(fb_switch):
                if elm is False:
                    self.classifiers[idx].cls = 0
        



    def forward(self, x):
        res = []
        h = None
        for i in range(self.nlays):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x)  # ReLU + Conv
            if self.maxpool[i] or i == self.nlays - 1:

                clf_id = len(res)
                # add classifier

                if self.ge is True:
                    x = gradient_rescale(x, 1.0 / (len(self.classifiers) - clf_id))

                if self.vanilla is True or len(res) == 0:
                    h, r = self.classifiers[clf_id](x, h) # hidden outputs, result outputs
                    res.append(r)
                else:
                    h, r = self.classifiers[clf_id](x, h)
                    res.append(r)

                if self.ge is True:
                    x = gradient_rescale(x, (len(self.classifiers) - clf_id - 1))

                if self.maxpool[i] is True:
                    x = self.maxpool2d(x)

        return res

