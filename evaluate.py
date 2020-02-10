'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import time
import os
import numpy as np
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

def main_cifar(args, gpunum=1, Tied=False, weightDecay=1e-3, nesterov=False):
    use_cuda = True # torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    batch_size = args.batch_size
    circles = args.circles
    backend = args.backend
    dataset_name = args.dataset_name
    lmbda = args.lmbda
    threshold = args.threshold
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
    

    model = torch.load(args.path)
    #print(model)
       
    
    # Define objective function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=lr, weight_decay=weightDecay, nesterov=nesterov)
      
    # Parallel computing
    if use_cuda:
        model.cuda()
        cudnn.benchmark = True
    
    # item() is a recent addition, so this helps with backward compatibility.
    def to_python_float(t):
        if hasattr(t, 'item'):
            return t.item()
        else:
            return t[0]
   
    # Testing
    def test(epoch):
        #nonlocal best_acc
        model.eval()
        test_loss = 0
        #correct = 0
        #total = 0
        corrects = np.zeros(100) # allocate large space 
        totals = np.zeros(100)
        exit_count = np.zeros(100)
        total_adaptive = 0
        correct_adaptive = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                if backend in ['modelE', 'modelF']:
                    outputs, errors = model(inputs)
                else:
                    outputs = model(inputs)
                #loss = criterion(outputs, targets)
                loss = 0.0
                for j in range(len(outputs)):
                    loss += criterion(outputs[j], targets)
                if backend in ['modelE', 'modelF']:
                    loss += lmbda * sum([torch.norm(errors[j]) for j in range(len(errors))]) / targets.shape[0]
            
                test_loss += to_python_float(loss.data)
                # multiple classifiers
                acc_str = ''
                for j in range(len(outputs)):
                    _, predicted = torch.max(outputs[j].data, 1)
                    totals[j] += targets.size(0)
                    corrects[j] += predicted.eq(targets.data).float().cpu().sum()

                    acc_str += '%.3f,'%(100.*corrects[j]/totals[j])

                # adaptive classifiers
                if epoch + 1 == max_epoch:
                    predicted_adaptive = torch.zeros(targets.shape[0]).long().to(targets.device)
                    for i in range(targets.shape[0]):
                        for j in range(len(outputs)):
                            confidence, idx = torch.topk(F.softmax(outputs[j][i]), k=1)
                            if confidence > threshold or j + 1 == len(outputs):
                                predicted_adaptive[i] = idx
                                exit_count[j] += 1
                                break
                    total_adaptive += targets.size(0)
                    correct_adaptive += predicted_adaptive.eq(targets.data).float().cpu().sum()
                    clf_exit_str = ' '.join(['%.3f' %(exit_count[i] / sum(exit_count[:len(outputs)])) for i in range(len(outputs))])
                else:
                    clf_exit_str = ''
                    correct_adaptive = 0
                    total_adaptive = 1e-5

                #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %s%%'
                #    % (test_loss/(batch_idx+1), acc_str))

                if batch_idx % 20 == 0:
                    #print('Batch: %d | Loss: %.3f | Acc: %s%%'%(batch_idx, test_loss/(batch_idx+1), acc_str))
                    print('Batch: %d | Loss: %.3f | Acc: %s%% | Adaptive Acc: %.3f%% | clf_exit: %s'%(batch_idx, test_loss/(batch_idx+1), acc_str, 100.*correct_adaptive / total_adaptive, clf_exit_str))


                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


        acc_str = ''
        for j in range(len(outputs)):
            acc_str += '%.3f,'%(100.*corrects[j]/totals[j])


        if epoch + 1 == max_epoch:
            clf_exit_str = ' '.join(['%.3f' %(exit_count[i] / sum(exit_count[:len(outputs)])) for i in range(len(outputs))])
        else:
            clf_exit_str = ''

        statstr = 'Testing: Epoch=%d | Loss: %.3f |  Acc: %s%% | Adaptive Acc:%.3f%% | clf_exit: %s ' \
                % (epoch, test_loss/(batch_idx+1), acc_str, 100.*correct_adaptive / total_adaptive, clf_exit_str)

        #statstr = 'Testing: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best_acc: %.3f' \
        #          % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, best_acc)
        print(statstr+'\n')
        
    # Test different circles:
    model.eval()
    model.dropout = 1.0
    model.adaptive = False
    for cls in range(args.circles + 1):
        # set cls
        print('circles:', cls)
        model.cls = cls
        for name, child in model.named_children():
            if name == 'classifiers':
                for clf in child.children():
                    #print(name, child)
                    clf.cls = cls
                    clf.dropout = 1.0
                    clf.adaptive = False

        start = time.time()
        test(max_epoch - 1)
        print('Time: ', time.time() - start)

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--circles', type=int, default=1)
    parser.add_argument('--lmbda', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--backend', type=str, required=True, choices=['modelA', 'modelB', 'modelC', 'modelD', 'modelE', 'modelF'])
    parser.add_argument('--dataset_name', type=str, required=True, choices=['cifar10', 'cifar100'])
    args = parser.parse_args()
    main_cifar(args)

