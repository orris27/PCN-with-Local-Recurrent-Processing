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
        from modelA import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles,Tied=Tied)
    elif backend == 'modelB':
        from modelB import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles,Tied=Tied)
    elif backend == 'modelC':
        from modelC import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge, fb=fb)
    
    elif backend == 'modelD':
        from modelD import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles, dropout=dropout, adaptive=adaptive, vanilla=vanilla, ge=ge, fb=fb)
    else:
        raise ValueError('backend: [modelA|modelB|modelC|modelD]')

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
        #correct = 0
        #total = 0
        corrects = np.zeros(100) # allocate large space 
        totals = np.zeros(100)
        exit_count = np.zeros(100)
        total_adaptive = 0
        correct_adaptive = 0
        
        training_setting = 'backend=%s | dataset=%s | adaptive=%d | batch_size=%d | epoch=%d | lr=%.1e | circles=%d | dropout=%.2f | step_all=%d | step_clf=%d | vanilla=%d | ge=%d | fb=%s' % (backend, dataset_name, adaptive, batch_size, epoch, optimizer.param_groups[0]['lr'], circles, dropout, step_all, step_clf, vanilla, ge, fb.replace(':', ''))
        statfile.write('\nTraining Setting: '+training_setting+'\n')
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            #if use_cuda:
            #    inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            #inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)

            #loss = criterion(outputs, targets)
            loss = 0.0
            for j in range(len(outputs)):
                loss += criterion(outputs[j], targets)

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

      
            if batch_idx % 20 == 0:
                print('Batch: %d | Loss: %.3f | Acc: %s%% | Adaptive Acc: %.3f%% | clf_exit: %s'%(batch_idx, train_loss/(batch_idx+1), acc_str, 100.*correct_adaptive / total_adaptive, clf_exit_str))
            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %s%%'
            #    % (train_loss/(batch_idx+1), acc_str))
            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        acc_str = ''
        for j in range(len(outputs)):
            acc_str += '%.3f,'%(100.*corrects[j]/totals[j])

        clf_exit_str = ' '.join(['%.3f' %(exit_count[i] / sum(exit_count[:len(outputs)])) for i in range(len(outputs))])

        statstr = 'Training: Epoch=%d | Loss: %.3f |  Acc: %s%% | Adaptive Acc:%.3f%% | clf_exit: %s ' \
                % (epoch, train_loss/(batch_idx+1), acc_str, 100.*correct_adaptive / total_adaptive, clf_exit_str)
        #statstr = 'Training: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best acc: %.3f' \
        #          % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total, best_acc)
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
                outputs = model(inputs)
                #loss = criterion(outputs, targets)
                loss = 0.0
                for j in range(len(outputs)):
                    loss += criterion(outputs[j], targets)
            
                test_loss += to_python_float(loss.data)
                # multiple classifiers
                acc_str = ''
                for j in range(len(outputs)):
                    _, predicted = torch.max(outputs[j].data, 1)
                    totals[j] += targets.size(0)
                    corrects[j] += predicted.eq(targets.data).float().cpu().sum()

                    acc_str += '%.3f,'%(100.*corrects[j]/totals[j])

                # adaptive classifiers
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


        clf_exit_str = ' '.join(['%.3f' %(exit_count[i] / sum(exit_count[:len(outputs)])) for i in range(len(outputs))])

        statstr = 'Testing: Epoch=%d | Loss: %.3f |  Acc: %s%% | Adaptive Acc:%.3f%% | clf_exit: %s ' \
                % (epoch, test_loss/(batch_idx+1), acc_str, 100.*correct_adaptive / total_adaptive, clf_exit_str)

        #statstr = 'Testing: Epoch=%d | Loss: %.3f |  Acc: %.3f%% (%d/%d) | best_acc: %.3f' \
        #          % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, best_acc)
        statfile.write(statstr+'\n')
        
        # Save checkpoint.
        #acc = 100.*correct/total
        #state = {
        #    'model': model.state_dict(),
        #    'acc': acc,
        #    'epoch': epoch,
        #}
        #torch.save(state, checkpointpath + modelname + '_last_ckpt.t7')
        #if acc >= best_acc:
            #print('Saving..')
            #torch.save(state, checkpointpath + modelname + '_best_ckpt.t7')
            #best_acc = acc
        
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
    setting = '%s_%s_adaptive%d_circles%d_dropout%.2f_all%dclf%d_vanilla%d_ge%d_fb%s' % (backend, dataset_name, adaptive, circles, dropout, step_all, step_clf, vanilla, ge, fb.replace(':', ''))
    torch.save(model, os.path.join('models', setting + '.pt'))

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
    parser.add_argument('--vanilla', type=int, default=0, help='no feed input from the previous classifiers') 
    parser.add_argument('--backend', type=str, required=True, choices=['modelA', 'modelB', 'modelC', 'modelD'])
    parser.add_argument('--dataset_name', type=str, required=True, choices=['cifar10', 'cifar100'])
    args = parser.parse_args()
    main_cifar(args)

