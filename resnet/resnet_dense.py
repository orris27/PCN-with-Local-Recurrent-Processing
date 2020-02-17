import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


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



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ClassifierModule(nn.Module):
    def __init__(self, in_channel_block, in_channel_clf, num_classes, adaptive, cls, dropout):
        super(ClassifierModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        #self.BN = nn.BatchNorm2d(in_channel_block)
        self.linear = nn.Linear(in_channel_block + in_channel_clf, num_classes)
        self.cls = cls # e.g.: 5
        if self.cls != 0:
            self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, num_classes))])
            self.linear_bw = nn.Linear(num_classes, in_channel_block + in_channel_clf)
        self.adaptive = adaptive
        self.dropout = dropout
        self.BN1d = nn.BatchNorm1d(num_classes)
        
    

    def forward(self, x_block, x_clf):
        out_block = F.avg_pool2d(x_block, x_block.size(-1))
        out_block = out_block.view(out_block.size(0), -1) # (batch_size, c_block)
        out_clf = x_clf # (batch_size, c_clf)

        if x_clf is None:
            out = out_block
        else:
            out = torch.cat([out_block, out_clf], dim=1)

        rep = self.linear(out) # representation

        if self.cls == 0 or (self.adaptive is True and self.training is False):
            #print('No feedback')
            pass
        elif self.dropout > 1.0 - 1e-12:
            b0 = F.relu(self.b0[0] + 1.0).expand_as(rep)
            for _ in range(self.cls):
                rep = self.linear(self.relu(out - self.linear_bw(rep))) * b0 + rep

        else:
            
            cls_local = random.randint(0, self.cls)
            b0 = F.relu(self.b0[0] + 1.0).expand_as(rep)
            for _ in range(cls_local):
                rep = self.linear(self.relu(out - self.linear_bw(rep))) * b0 + rep

        rep = self.BN1d(rep)

        # no bypass

        return rep



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cls=0, dropout=1.0, adaptive=False, vanilla=False, ge=False):
        super(ResNet, self).__init__()

        # early-exiting setting
        self.cls = cls
        self.adaptive = adaptive
        self.dropout = dropout
        self.vanilla = vanilla
        self.ge = ge


        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        #self.layers = []
        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(block, 16, num_blocks[0], stride=1))
        self.layers.append(self._make_layer(block, 32, num_blocks[1], stride=2))
        self.layers.append(self._make_layer(block, 64, num_blocks[2], stride=2))

        #self.classifiers = []
        self.classifiers = nn.ModuleList()
        self.classifiers.append(ClassifierModule(16, 0, num_classes, self.adaptive, self.cls, self.dropout))
        self.classifiers.append(ClassifierModule(32, num_classes, num_classes, self.adaptive, self.cls, self.dropout))
        self.classifiers.append(ClassifierModule(64, 2 * num_classes, num_classes, self.adaptive, self.cls, self.dropout))

        #self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        res = []
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(3):
            out = self.layers[i](out)
            clf_id = len(res)

            if self.ge is True:
                out = gradient_rescale(out, 1.0 / (len(self.classifiers) - clf_id))

            if self.vanilla is True or len(res) == 0:
                r = self.classifiers[clf_id](out, None) # representation, hidden outputs
                res.append(r)
            else:
                r = self.classifiers[clf_id](out, torch.cat(res, dim=1))
                res.append(r)

            if self.ge is True:
                out = gradient_rescale(out, (len(self.classifiers) - clf_id - 1))

        return res


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56(num_classes, cls, dropout, adaptive, vanilla, ge):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, cls, dropout, adaptive, vanilla, ge)


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

