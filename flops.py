import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------ # 
class FConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(FConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FConv2d, self).forward(x)
        output_area = output.size(-1)*output.size(-2)
        filter_area = np.prod(self.kernel_size)
        self.num_ops += self.in_channels*self.out_channels*filter_area*output_area
        return output

class FLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(FLinear, self).__init__(in_features, out_features, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FLinear, self).forward(x)
        self.num_ops += self.in_features*self.out_features
        return output

def count_flops(model, reset=True):
    op_count = 0
    for m in model.modules():
        if hasattr(m, 'num_ops'):
            op_count += m.num_ops
            if reset: # count and reset to 0
                m.num_ops = 0

    return op_count

# replace all nn.Conv and nn.Linear layers with layers that count flops
nn.Conv2d = FConv2d
nn.Linear = FLinear

# ------------------------------------------------------------------ # 
device = torch.device('cpu')
x = torch.randn(1, 3, 32, 32).to(device) # dummy inputs

# PCN

print('PCN')
for num_classes in [10, 100]:
    print('num_classes:', num_classes)
    for circles in [0, 1, 2]:
        print('circles:', circles)
        from pcn.prednet import PredNetBpD
        model = PredNetBpD(num_classes=num_classes, cls=circles)

        params = sum([w.numel() for name, w in model.named_parameters()])
        y_predicted = model.forward(x)

        flops = count_flops(model)
        print('flops: %d | params: %d' % (flops, params))

#print(model)
print('\n'*3)


print('PCN-E')
for num_classes in [10, 100]:
    print('num_classes:', num_classes)
    for circles in [0, 1, 2]:
        print('circles:', circles)
        from pcn.prednetE import PredNetBpD
        model = PredNetBpD(num_classes=num_classes, cls=circles)

        params = sum([w.numel() for name, w in model.named_parameters()])
        y_predicted = model.forward(x)

        flops = count_flops(model)
        print('flops: %d | params: %d' % (flops, params))

#print(model)
print('\n'*3)


print('PCN_h')
for num_classes in [10, 100]:
    print('num_classes:', num_classes)
    for circles in [0, 1, 2]:
        print('circles:', circles)
        from pcn.prednet_h import PredNetBpD
        model = PredNetBpD(num_classes=num_classes, cls=circles).eval()
    

        params = sum([w.numel() for name, w in model.named_parameters()])
        y_predicted = model.forward(x)

        flops = count_flops(model)
        print('flops: %d | params: %d' % (flops, params))

#print(model)
print('\n'*3)


print('resnet_h_bl')
for num_classes in [10, 100]:
    print('num_classes:', num_classes)
    for circles in [0]:
        print('circles:', circles)
        from resnet.resnet_h_bl import resnet56
        model = resnet56(num_classes=num_classes).eval()

        params = sum([w.numel() for name, w in model.named_parameters()])
        y_predicted = model.forward(x)

        flops = count_flops(model)
        print('flops: %d | params: %d' % (flops, params))

#print(model)
print('\n'*3)

print('resnet_bl')
for num_classes in [10, 100]:
    print('num_classes:', num_classes)
    for circles in [0]:
        print('circles:', circles)
        from resnet.resnet_bl import resnet56
        model = resnet56(num_classes=num_classes).eval()

        params = sum([w.numel() for name, w in model.named_parameters()])
        y_predicted = model.forward(x)

        flops = count_flops(model)
        print('flops: %d | params: %d' % (flops, params))

#print(model)
print('\n'*3)



print('modelC_dp2_bl')
for num_classes in [10, 100]:
    print('num_classes:', num_classes)
    for circles in [0]:
        print('circles:', circles)
        from pcn.modelC_dp2_bl import PredNetBpD
        model = PredNetBpD(num_classes=num_classes).eval()

        params = sum([w.numel() for name, w in model.named_parameters()])
        y_predicted = model.forward(x)

        flops = count_flops(model)
        print('flops: %d | params: %d' % (flops, params))

#print(model)
print('\n'*3)









# Ours
#print('Ours')
#f = []
#for num_classes in [10, 100]:
#    print('num_classes:', num_classes)
#    for circles in [0, 1, 2]:
#        print('circles:', circles)
#        from modelC_h_dp2 import PredNetBpD
#        model = PredNetBpD(num_classes=num_classes,cls=circles, dropout=1.0, adaptive=False, vanilla=False, ge=0, fb='1:1:1')
#        model = model.eval()
#
#        params = sum([w.numel() for name, w in model.named_parameters()])
#        y_predicted = model.forward(x)
#
#        flops = count_flops(model)
#        f.append(flops)
#        print('flops: %d | params: %d' % (flops, params))
#    print(f[1] - f[0], f[2] - f[1])
#print(model)


#print('Ours')
#f = []
#for num_classes in [10, 100]:
#    print('num_classes:', num_classes)
#    for circles in [0, 1, 2]:
#        print('circles:', circles)
#        from resnet.resnet import resnet56
#        model = resnet56(num_classes=num_classes,cls=circles, dropout=1.0, adaptive=False, vanilla=False, ge=0)
#        model = model.eval()
#
#        params = sum([w.numel() for name, w in model.named_parameters()])
#        y_predicted = model.forward(x)
#
#        flops = count_flops(model)
#        f.append(flops)
#        print('flops: %d | params: %d' % (flops, params))
#    print(f[1] - f[0], f[2] - f[1])
#print(model)


print('modelG')
f = []
for num_classes in [10, 100]:
    print('num_classes:', num_classes)
    for circles in [0, 1, 2]:
        print('circles:', circles)
        from pcn.modelG import PredNetBpD
        model = PredNetBpD(num_classes=num_classes,cls=circles, ge=False)
        model = model.eval()

        params = sum([w.numel() for name, w in model.named_parameters()])
        y_predicted = model.forward(x)

        flops = count_flops(model)
        f.append(flops)
        print('flops: %d | params: %d' % (flops, params))
    print(f[1] - f[0], f[2] - f[1])
print(model)
