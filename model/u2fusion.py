import sys
import torch
import torch.nn as nn
import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# tool functions
def get_img_seq(img_seq_dir):
    img_seq = []
    for root, _, fnames in sorted(os.walk(img_seq_dir)):
        for fname in sorted(fnames):
            if any(fname.endswith(ext) for ext in args.ext):
                img_name = os.path.join(root, fname)
                img_seq.append(cv2.imread(img_name))
    return img_seq


def features_grad(features):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads


# network functions
def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type='prelu', slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(negative_slope=slope, inplace=True)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!' % act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % norm_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,
              act_type='prelu', norm_type='bn', pad_type='zero'):
    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias)

    act = activation(act_type) if act_type else None
    n = norm(out_channels, norm_type) if norm_type else None
    return sequential(p, conv, n, act)


class DenseLayer(nn.Module):
    def __init__(self, num_channels, growth):
        super(DenseLayer, self).__init__()
        self.conv = ConvBlock(num_channels, growth, kernel_size=3, act_type='lrelu', norm_type=None)

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)
        return out


class DenseNet(nn.Module):
    def __init__(self, in_channels=3, num_features=44, growth=44, num_layers=5):
        super(DenseNet, self).__init__()
        self.num_channels = 2 * in_channels
        self.num_features = num_features
        self.growth = growth
        modules = []
        self.conv_1 = ConvBlock(self.num_channels, self.num_features, kernel_size=3, act_type='lrelu', norm_type=None)
        for i in range(num_layers):
            modules.append(DenseLayer(self.num_features, self.growth))
            self.num_features += self.growth
        self.dense_layers = nn.Sequential(*modules)
        self.sub = nn.Sequential(ConvBlock(self.num_features, 128, kernel_size=3, act_type='lrelu', norm_type=None),
                                 ConvBlock(128, 64, kernel_size=3, act_type='lrelu', norm_type=None),
                                 ConvBlock(64, 32, kernel_size=3, act_type='lrelu', norm_type=None),
                                 nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1),
                                 nn.Tanh())

    def forward(self, x_over, x_under):
        x = torch.cat((x_over, x_under), dim=1)
        x = self.conv_1(x)
        x = self.dense_layers(x)
        x = self.sub(x)
        return x


    def forward(self, x_over, x_under):
        x = torch.cat((x_over, x_under), dim=1)
        x = self.conv_1(x)
        x = self.dense_layers(x)
        x = self.sub(x)
        return x


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    from utils import measure_throughput
    
    x1 = torch.randn(1, 3, 256, 256).cuda()
    x2 = torch.randn(1, 3, 256, 256).cuda()
    print(flop_count_table(
        FlopCountAnalysis(DenseNet(3, 44, 44, 5).cuda(), (x1, x2))
    ))
    
    measure_throughput(DenseNet(3, 44, 44, 5).cuda(), [(3, 256, 256), (3, 256, 256)], 32)