import torch.nn as nn
import torch.nn.init as init


def he_normal(net, a=0, mode='fan_in'):
    for m in net.modules():
        if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear)):
            init.kaiming_normal(m.weight, a=a, mode=mode)
            if m.bias is not None:
                init.constant(m.bias, 0.)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            init.constant(m.weight, 1.)
            if m.bias is not None:
                init.constant(m.bias, 0.)
        else:
            pass

