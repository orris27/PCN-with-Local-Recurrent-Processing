'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
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
    adaptive = bool(args.adaptive)
    threshold = args.threshold
    max_epoch = args.max_epoch
    dropout = args.dropout
    step_all, step_clf = args.step_all, args.step_clf
    vanilla = bool(args.vanilla)
    ge = bool(args.ge)
    lmbda = args.lmbda
    fb = args.fb
    root = './'
    rep = 1
    lr = 0.01

    assert vanilla is False or circles == 0 # "vanilla is True and circles != 0" is not valid
    
    modelname = 'PredNetBpD' +'_'+str(circles)+'CLS_'+str(nesterov)+'Nes_'+str(weightDecay)+'WD_'+str(Tied)+'TIED_'+str(rep)+'REP'
    
    # clearn folder
    checkpointpath = root+'checkpoint/'
    logpath = root+'log/'
    if not os.path.isdir(checkpointpath):
        os.mkdir(checkpointpath)
    if not os.path.isdir(logpath):
        os.mkdir(logpath)
    while(os.path.isfile(logpath+'training_stats_'+modelname+'.txt')):
        rep += 1
        modelname = 'PredNetBpD'+'_'+str(circles)+'CLS_'+str(nesterov)+'Nes_'+str(weightDecay)+'WD_'+str(Tied)+'TIED_'+str(rep)+'REP'
        
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
    
    if backend == 'modelA':
        from pcn.modelA import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles,Tied=Tied)
    elif backend == 'modelB':
        from pcn.modelB import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles,Tied=Tied)
    elif backend == 'modelC':
        from pcn.modelC import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge, fb=fb)
    elif backend == 'modelC_dp2':
        from pcn.modelC_dp2 import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge, fb=fb)
    elif backend == 'modelC_h_dp2':
        from pcn.modelC_h_dp2 import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge, fb=fb)
    elif backend == 'modelD':
        from pcn.modelD import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge, fb=fb)
    elif backend == 'modelE':
        from pcn.modelE import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge, fb=fb)
    elif backend == 'modelE_dp2':
        from pcn.modelE_dp2 import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge, fb=fb)
    elif backend == 'modelF':
        from pcn.modelF import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge, fb=fb)
    elif backend == 'resnet56':
        from resnet.resnet import resnet56
        model = resnet56(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge)
    elif backend == 'resnet56_h':
        from resnet.resnet_h import resnet56
        model = resnet56(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge)
    elif backend == 'resnet56_dense':
        from resnet.resnet_dense import resnet56
        model = resnet56(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge)
    else:
        raise ValueError

    print(model)
       
    
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
   
   # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        corrects = np.zeros(100) # allocate large space 
        totals = np.zeros(100)
        exit_count = np.zeros(100)
        total_adaptive = 0
        correct_adaptive = 0
        
        training_setting = str(args)
        statfile.write('\nTraining Setting: '+training_setting+'\n')
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            if backend in ['modelE', 'modelE_dp2', 'modelF']:
                outputs, errors = model(inputs)
            else:
                outputs = model(inputs)

            #loss = criterion(outputs, targets)
            loss = 0.0
            for j in range(len(outputs)):
                loss += criterion(outputs[j], targets)
            if backend in ['modelE', 'modelE_dp2', 'modelF']:
                loss += lmbda * sum([torch.norm(errors[j]) for j in range(len(errors)) if errors[j] is not None]) / targets.shape[0]

            loss.backward()
            optimizer.step()
    
            train_loss += to_python_float(loss.data)

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

      
            if batch_idx % 20 == 0:
                print('Batch: %d | Loss: %.3f | Acc: %s%% | Adaptive Acc: %.3f%% | clf_exit: %s'%(batch_idx, train_loss/(batch_idx+1), acc_str, 100.*correct_adaptive / total_adaptive, clf_exit_str))

        acc_str = ''
        for j in range(len(outputs)):
            acc_str += '%.3f,'%(100.*corrects[j]/totals[j])

        if epoch + 1 == max_epoch:
            clf_exit_str = ' '.join(['%.3f' %(exit_count[i] / sum(exit_count[:len(outputs)])) for i in range(len(outputs))])
        else:
            clf_exit_str = ''

        statstr = 'Training: Epoch=%d | Loss: %.3f |  Acc: %s%% | Adaptive Acc:%.3f%% | clf_exit: %s ' \
                % (epoch, train_loss/(batch_idx+1), acc_str, 100.*correct_adaptive / total_adaptive, clf_exit_str)
        statfile.write(statstr+'\n')
    
    
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
                if backend in ['modelE', 'modelE_dp2', 'modelF']:
                    outputs, errors = model(inputs)
                else:
                    outputs = model(inputs)
                #loss = criterion(outputs, targets)
                loss = 0.0
                for j in range(len(outputs)):
                    loss += criterion(outputs[j], targets)
                if backend in ['modelE', 'modelE_dp2', 'modelF']:
                    loss += lmbda * sum([torch.norm(errors[j]) for j in range(len(errors)) if errors[j] is not None]) / targets.shape[0]
            
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

                if batch_idx % 20 == 0:
                    print('Batch: %d | Loss: %.3f | Acc: %s%% | Adaptive Acc: %.3f%% | clf_exit: %s'%(batch_idx, test_loss/(batch_idx+1), acc_str, 100.*correct_adaptive / total_adaptive, clf_exit_str))



        acc_str = ''
        for j in range(len(outputs)):
            acc_str += '%.3f,'%(100.*corrects[j]/totals[j])


        if epoch + 1 == max_epoch:
            clf_exit_str = ' '.join(['%.3f' %(exit_count[i] / sum(exit_count[:len(outputs)])) for i in range(len(outputs))])
        else:
            clf_exit_str = ''

        statstr = 'Testing: Epoch=%d | Loss: %.3f |  Acc: %s%% | Adaptive Acc:%.3f%% | clf_exit: %s ' \
                % (epoch, test_loss/(batch_idx+1), acc_str, 100.*correct_adaptive / total_adaptive, clf_exit_str)

        statfile.write(statstr+'\n')
        
        
    # Set adaptive learning rates
    def decrease_learning_rate():
        """Decay the previous learning rate by 10"""
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

    
    for epoch in range(max_epoch):
        statfile = open(logpath+'training_stats_'+modelname+'.txt', 'a+')
        if epoch==150 or epoch==225 or epoch == 262:
            decrease_learning_rate()       
        if step_all != 0 and step_clf != 0:
            if epoch % (step_all + step_clf) < step_all:
                print('Train all parameters')
                model.requires_grad_(True)
            else:
                print('Train classifier parameters')
                model.requires_grad_(False)
                for clf in model.classifiers:
                    clf.requires_grad_(True)
        train(epoch)
        test(epoch)
    os.makedirs('models/', exist_ok=True)
    setting = '%s_%s_adaptive%d_circles%d_dropout%.2f_all%dclf%d_vanilla%d_ge%d_fb%s_lmbda%.4f' % (backend, dataset_name, adaptive, circles, dropout, step_all, step_clf, vanilla, ge, fb.replace(':', ''), lmbda)
    print('model is save as %s'%(os.path.join('models', setting + '.pt')))
    torch.save(model.state_dict(), os.path.join('models', setting + '.pt'))


    # Test different circles:
    print('Evaluate with different circles:')

    model.eval()
    model.dropout = 1.0
    model.adaptive = False
    statfile.write('\n')
    for cls in range(args.circles + 1):
        # set cls
        statfile.write('circles: %d \n'%(cls))
        model.cls = cls
        for name, child in model.named_children():
            if name == 'classifiers':
                for clf in child.children():
                    #print(name, child)
                    clf.cls = cls
                    clf.dropout = 1.0
                    clf.adaptive = False

        test(max_epoch - 1)        

    with open(logpath+'training_stats_'+modelname+'.txt', 'r') as statfile:
        print('\n' * 5)
        for line in statfile.readlines():
            print(line.strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--circles', type=int, default=1)
    parser.add_argument('--adaptive', type=int, default=0)
    parser.add_argument('--ge', type=int, default=0, help='gradient equilibrium')
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--fb', type=str, default='1:1:1')
    parser.add_argument('--step_all', type=int, default=0) # 15
    parser.add_argument('--step_clf', type=int, default=0) # 10
    parser.add_argument('--lmbda', type=float, default=0.0)
    parser.add_argument('--vanilla', type=int, default=0, help='no feed input from the previous classifiers') 
    parser.add_argument('--backend', type=str, required=True, choices=['modelA', 'modelB', 'modelC', 'modelC_dp2', 'modelC_h_dp2', 'modelD', 'modelE', 'modelE_dp2',  'modelF', 'resnet56', 'resnet56_h', 'resnet56_dense'])
    parser.add_argument('--dataset_name', type=str, required=True, choices=['cifar10', 'cifar100'])
    args = parser.parse_args()
    main_cifar(args)

