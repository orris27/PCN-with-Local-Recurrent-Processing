'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd

class KDLoss(nn.Module):
    def __init__(self, T, gamma, num_classifiers):
        super(KDLoss, self).__init__()

        self.kld_loss = nn.KLDivLoss().cuda()
        self.ce_loss = nn.CrossEntropyLoss().cuda()
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

        self.T = T
        self.gamma = gamma
        self.num_classifiers = num_classifiers

    def loss_fn_kd(self, outputs, targets, soft_targets):
        loss = self.ce_loss(outputs[-1], targets)
        T = self.T
        for i in range(self.num_classifiers - 1):
            _ce = (1. - self.gamma) * self.ce_loss(outputs[i], targets)
            _kld = self.kld_loss(self.log_softmax(outputs[i] / T), self.softmax(soft_targets.detach() / T)) * self.gamma * T * T
            loss = loss + _ce + _kld
        return loss

def ee_loss(outputs, targets, scores, flops_list, lmbda):
    criterion = nn.CrossEntropyLoss()

    ce = 0.0 
    Ys = []
    Ys.append(outputs[-1])
    # loss for cross entropy
    ce += criterion(Ys[-1], targets)
    for i in reversed(range(2)):
        Ys.append(scores[i] * outputs[i] + (1 - scores[i]) * Ys[-1])
        ce += criterion(Ys[-1], targets)

    # loss for regularizer item
    flops_list = [0.0] + [flops / max(flops_list) for flops in flops_list] 
    Cs = []
    Cs.append(flops_list[-1])
    reg = 0.0
    reg += Cs[-1]
    for i in reversed(range(0, len(outputs) - 1)): #  1, 0
        #Cs.append(scores[i] * (flops_list[i + 1] - flops_list[i]) + (1 - scores[i]) * (Cs[-1] - flops_list[i + 1]))
        #Cs.append(scores[i] * (flops_list[i + 1]) + (1 - scores[i]) * (Cs[-1] - flops_list[i + 1]))
        Cs.append(scores[i] * (flops_list[i + 1]) + (1 - scores[i]) * (Cs[-1]))
        reg += torch.mean(Cs[-1])
    #return ce + lmbda * reg + criterion(outputs[0], targets) + criterion(outputs[1], targets)
    return ce + lmbda * reg



#def get_mean_and_std(dataset):
#    '''Compute the mean and std value of dataset.'''
#    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
#    mean = torch.zeros(3)
#    std = torch.zeros(3)
#    print('==> Computing mean and std..')
#    for inputs, targets in dataloader:
#        for i in range(3):
#            mean[i] += inputs[:,i,:,:].mean()
#            std[i] += inputs[:,i,:,:].std()
#    mean.div_(len(dataset))
#    std.div_(len(dataset))
#    return mean, std
#
#def init_params(net):
#    '''Init layer parameters.'''
#    for m in net.modules():
#        if isinstance(m, nn.Conv2d):
#            init.kaiming_normal(m.weight, mode='fan_out')
#            if m.bias:
#                init.constant(m.bias, 0)
#        elif isinstance(m, nn.BatchNorm2d):
#            init.constant(m.weight, 1)
#            init.constant(m.bias, 0)
#        elif isinstance(m, nn.Linear):
#            init.normal(m.weight, std=1e-3)
#            if m.bias:
#                init.constant(m.bias, 0)
#
#
#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)
#
#TOTAL_BAR_LENGTH = 65.
#last_time = time.time()
#begin_time = last_time
#def progress_bar(current, total, msg=None):
#    global last_time, begin_time
#    if current == 0:
#        begin_time = time.time()  # Reset for new bar.
#
#    cur_len = int(TOTAL_BAR_LENGTH*current/total)
#    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
#
#    sys.stdout.write(' [')
#    for i in range(cur_len):
#        sys.stdout.write('=')
#    sys.stdout.write('>')
#    for i in range(rest_len):
#        sys.stdout.write('.')
#    sys.stdout.write(']')
#
#    cur_time = time.time()
#    step_time = cur_time - last_time
#    last_time = cur_time
#    tot_time = cur_time - begin_time
#
#    L = []
#    L.append('  Step: %s' % format_time(step_time))
#    L.append(' | Tot: %s' % format_time(tot_time))
#    if msg:
#        L.append(' | ' + msg)
#
#    msg = ''.join(L)
#    sys.stdout.write(msg)
#    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#        sys.stdout.write(' ')
#
#    # Go back to the center of the bar.
#    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#        sys.stdout.write('\b')
#    sys.stdout.write(' %d/%d ' % (current+1, total))
#
#    if current < total-1:
#        sys.stdout.write('\r')
#    else:
#        sys.stdout.write('\n')
#    sys.stdout.flush()
#
#def format_time(seconds):
#    days = int(seconds / 3600/24)
#    seconds = seconds - days*3600*24
#    hours = int(seconds / 3600)
#    seconds = seconds - hours*3600
#    minutes = int(seconds / 60)
#    seconds = seconds - minutes*60
#    secondsf = int(seconds)
#    seconds = seconds - secondsf
#    millis = int(seconds*1000)
#
#    f = ''
#    i = 1
#    if days > 0:
#        f += str(days) + 'D'
#        i += 1
#    if hours > 0 and i <= 2:
#        f += str(hours) + 'h'
#        i += 1
#    if minutes > 0 and i <= 2:
#        f += str(minutes) + 'm'
#        i += 1
#    if secondsf > 0 and i <= 2:
#        f += str(secondsf) + 's'
#        i += 1
#    if millis > 0 and i <= 2:
#        f += str(millis) + 'ms'
#        i += 1
#    if f == '':
#        f = '0ms'
#    return f
