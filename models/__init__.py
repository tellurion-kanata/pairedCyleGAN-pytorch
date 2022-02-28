from .basemodel import *
from .networks import *
from torch.nn import init

import torch.cuda


__all__ = [
    'BaseModel', 'define_D', 'loss', 'init_net', 'DeepResidualNetwork'
]

""" GAN Loss, parallel training and initialization are from Jun-Yan Zhu et al.'s implementation 
Github: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
"""

""" Applying zero normalization to generator. """
def init_net(net, init_gain=0.02, init_type='normal', gpus=[]):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    if len(gpus) > 0:
        assert(torch.cuda.is_available())
        net = nn.DataParallel(net, gpus).to(gpus[0])
    net.apply(init_func)
    return net


def define_D(input_channels, n_layers=3, ndf=64, downsample=False, spec_norm=False, use_bias=True, gpus=[]):
    net = CustomDiscriminator(input_channels, n_layers, ndf, downsample, spec_norm, use_bias)
    net = init_net(net, init_type='kaiming', gpus=gpus)
    return net
