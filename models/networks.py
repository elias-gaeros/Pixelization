import torch
import torch.nn as nn
from torch.nn import init
import safetensors.torch as st
import functools
from .c2pGen import C2PGen, AliasNet
from .p2cGen import P2CGen


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    return net.apply(init_func)  # apply the initialization function <init_func>


def to_device(net, gpu_ids=[]):
    """register CPU/GPU device (with multi-GPU support)
    Parameters:
        net (network)      -- the network to be initialized
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 1:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        return torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    elif len(gpu_ids) == 1:
        assert(torch.cuda.is_available())
        return net.to(torch.device(gpu_ids[0]))
    else:
        return net


def define_G(input_nc, output_nc, ngf, netG, init_type='normal', init_gain=0.02, gpu_ids=[], init=True):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    """
    net = None

    if netG == 'c2pGen':  #                     style_dim  mlp_dim
        net = C2PGen(input_nc, output_nc, ngf, 2, 4, 256, 256, activ='relu', pad_type='reflect')
    elif netG == 'p2cGen':
        net = P2CGen(input_nc, output_nc, ngf, 2, 3, activ='relu', pad_type='reflect')
    elif netG == 'antialias':
        net = AliasNet(input_nc, output_nc, ngf, 2, 3, activ='relu', pad_type='reflect')
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    net = to_device(net, gpu_ids)

    if init:
        net = init_weights(net, init_type, init_gain=init_gain)
        if netG == 'c2pGen':
            features = net.PBEnc.vgg
            device = str(next(features.parameters()).device)
            state_dict = st.load_file('./pixelart_vgg19.safetensors', device=str(device))
            features.load_state_dict({
                k.removeprefix('features.'): v
                for k, v in state_dict.items()
                if k.startswith('features.')
            })

    return net

