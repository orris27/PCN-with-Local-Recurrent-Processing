'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import time
import os
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
#from utils import progress_bar
from torch.autograd import Variable


class Combination(nn.Module):
    def __init__(self, channel):
        super(Combination, self).__init__()
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(channel)
        self.linear = nn.Linear(channel, 8)

    def forward(self, x):
        out = F.avg_pool2d(self.relu(self.BN(x)), x.size(-1))
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        return out


def main_cifar(args, gpunum=1, Tied=False, weightDecay=1e-3, nesterov=False):
    use_cuda = True # torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    batch_size = args.batch_size
    circles = args.circles
    backend = args.backend
    dataset_name = args.dataset_name
    lmbda = args.lmbda
    threshold = args.threshold
    mode = args.mode
    flops = list(map(int, args.flops.split(':')))
    max_epoch = 100
    root = './'
    rep = 1
    lr = 0.01

    
    modelname = 'PredNetBpD' +'_'+str(circles)+'CLS_'+str(nesterov)+'Nes_'+str(weightDecay)+'WD_'+str(Tied)+'TIED_'+str(rep)+'REP'
    
    # Data
    print('==> Preparing data..')
    data_dir = '../../datasets/torchvision'
    if dataset_name == 'cifar10':
        print('Dataset: CIFAR10')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        num_classes = 10
    elif dataset_name == 'cifar100':
        print('Dataset: CIFAR100')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        num_classes = 100
    else:
        raise ValueError('dataset_name: [cifar10|cifar100]')
    
    # Model
    print('==> Building model..')
    

    #model = torch.load(args.path)
    from pcn.modelC_dp2 import PredNetBpD
    model = PredNetBpD(num_classes=100,cls=5, dropout=1.0, adaptive=False, vanilla=False, ge=1, fb='1:1:1')
    model.load_state_dict(torch.load(args.path))
    model = model.eval()

    print(model)

    model_c = Combination(128)
    print(model_c)

    # Define objective function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_c.parameters(), momentum=0.9, lr=lr, weight_decay=weightDecay, nesterov=nesterov)
      
    # Parallel computing
    if use_cuda:
        model.cuda()
        model_c.cuda()
        cudnn.benchmark = True

    # Testing
    def train(model_c, epoch):
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            y = torch.zeros(inputs.size(0)).long()
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                y = y.cuda()
            with torch.no_grad():
                # X
                #features = model.xxx(inputs)
                features = model.BNs[0](inputs)
                features = model.PcConvs[0](features)
                features = model.BNs[1](features)
                features = model.PcConvs[1](features)
                features = model.BNs[2](features)
                features = model.PcConvs[2](features) # (B, 128, 32, 32)
                features = model.maxpool2d(features) # (B, 128, 16, 16)
                # Y
                outputs = model(inputs)
                for j in range(len(outputs)):
                    _, predicted = torch.max(outputs[j].data, 1)
                    true_idx = predicted.eq(targets.data)
                    y[true_idx] += 2 ** j

            y_predicted = model_c(features)


            loss = criterion(y_predicted, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(y_predicted.data, 1)
            total += y.size(0)
            correct += predicted.eq(y.data).float().cpu().sum()

            if batch_idx % 20 == 0:
                print('Batch: %d | Loss: %.3f | Acc: %.3f%%'%(batch_idx, loss, correct/total))


    def test(model_c, epoch):
        model_c = model_c.eval()
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            y = torch.zeros(inputs.size(0)).long()
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                y = y.cuda()
            with torch.no_grad():
                # X
                features = model.BNs[0](inputs)
                features = model.PcConvs[0](features)
                features = model.BNs[1](features)
                features = model.PcConvs[1](features)
                features = model.BNs[2](features)
                features = model.PcConvs[2](features) # (B, 128, 32, 32)
                features = model.maxpool2d(features) # (B, 128, 16, 16)
                # Y
                outputs = model(inputs)
                for j in range(len(outputs)):
                    _, predicted = torch.max(outputs[j].data, 1)
                    true_idx = predicted.eq(targets.data)
                    y[true_idx] += 2 ** j

                y_predicted = model_c(features)

            _, predicted = torch.max(y_predicted.data, 1)
            total += y.size(0)
            correct += predicted.eq(y.data).float().cpu().sum()

            if batch_idx % 20 == 0:
                print('Batch: %d | Acc: %.3f%%'%(batch_idx, correct/total))
        model_c = model_c.train()



    for epoch in range(max_epoch):
        train(model_c, epoch)
        test(model_c, epoch)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)

    parser.add_argument('--flops', type=str, default='0:0:0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--circles', type=int, default=1)
    parser.add_argument('--lmbda', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--mode', type=str, required=True, choices=['threshold', 'random', 'first_classifier'])
    parser.add_argument('--backend', type=str, required=True, choices=['modelA', 'modelB', 'modelC', 'modelC_h_dp2', 'modelD', 'modelE', 'modelF', 'modelC_dp2'])
    parser.add_argument('--dataset_name', type=str, required=True, choices=['cifar10', 'cifar100'])
    args = parser.parse_args()
    main_cifar(args)

