"""
Author: Jieyi Zhu, Zihan Cao
Date: 2024-11-06

UESTC copyright (c) 2024

"""

import torch
import torch.nn as nn
import math
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm
from torchdeq.dropout import VariationalDropout2d, reset_dropout
from torchdeq.core import DEQSliced
from torchdeq.utils.layer_utils import SpeedyMDEQWrapper, MDEQWrapper

from utils import easy_logger

logger = easy_logger(func_name='UnfoldingDEQ')

import sys
sys.path.append('./')

# from Unet import Unet
from model.FusionNet import FusionNet


#* ==================================================================================
#* UNet implemented 

class Encoding_Block(torch.nn.Module):
    def __init__(self, c_in):
        super(Encoding_Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=3 // 2)

        self.act = torch.nn.PReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):

        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        down = self.act(self.conv5(f_e))
        return f_e, down


class Encoding_Block_End(torch.nn.Module):
    def __init__(self, c_in=64):
        super(Encoding_Block_End, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=3 // 2)
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):
        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        return f_e


class Decoding_Block(torch.nn.Module):
    def __init__(self, c_in):
        super(Decoding_Block, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)
        self.up = torch.nn.ConvTranspose2d(c_in, 64, kernel_size=3, stride=2, padding=3 // 2)

        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input, map):

        up = self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]])
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))

        out3 = self.conv3(out2)

        return out3


class Feature_Decoding_End(torch.nn.Module):
    def __init__(self, c_out):
        super(Feature_Decoding_End, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=3, padding=3 // 2)
        self.batch = 1
        self.up = torch.nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=3 // 2)
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input, map):

        up = self.up(input, output_size=[input.shape[0], input.shape[1], map.shape[2], map.shape[3]])
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))

        out3 = self.conv3(out2)

        return out3


class ResConv(torch.nn.Module):
    def __init__(self, cin, mid, cout, kernel_size=3):
        super(ResConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=cin, out_channels=mid, kernel_size=kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(in_channels=mid, out_channels=cout, kernel_size=1, padding="same")
        self.relu = torch.nn.PReLU()

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        out = x + res
        return out

class Unet(torch.nn.Module):
    def __init__(self, cin):
        super(Unet, self).__init__()

        self.Encoding_block1 = Encoding_Block(cin)
        self.Encoding_block2 = Encoding_Block(32)
        self.Encoding_block3 = Encoding_Block(32)
        self.Encoding_block4 = Encoding_Block(32)
        self.Encoding_block_end = Encoding_Block_End(32)

        self.Decoding_block1 = Decoding_Block(64)
        self.Decoding_block2 = Decoding_Block(256)
        self.Decoding_block3 = Decoding_Block(256)
        self.Decoding_block_End = Feature_Decoding_End(cin)

        # self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)

        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block_End(decode1, encode0)

        return decode0, encode0


#* ==================================================================================
#* Unfolding Model for Pansharpening


class Unfolding(nn.Module):
    def __init__(self, num_spectral, num_channel, factor, lambda1, lambda2):
        super().__init__()

        self.delta_0 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_0 = torch.nn.Parameter(torch.tensor(0.9))
        # self.lambda1 = torch.nn.Parameter(torch.tensor(lambda1))
        # self.lambda2 = torch.nn.Parameter(torch.tensor(lambda2))
        
        # =================== convs ===================
        KERNEL_SIZE = 3  # set as MoGDCN
        self.conv_downsample = torch.nn.Conv2d(in_channels=num_spectral, out_channels=num_spectral, kernel_size=KERNEL_SIZE, stride=factor,
                                               padding=KERNEL_SIZE // 2)
        self.conv_upsample = torch.nn.ConvTranspose2d(in_channels=num_spectral, out_channels=num_spectral, kernel_size=KERNEL_SIZE, stride=factor,
                                                      padding=KERNEL_SIZE // 2)

        self.conv_topan = torch.nn.Conv2d(in_channels=num_spectral, out_channels=1, kernel_size=3, stride=1, padding=3 // 2)
        self.conv_tolms = torch.nn.Conv2d(in_channels=1, out_channels=num_spectral, kernel_size=3, stride=1, padding=3 // 2)

        # =================== x_net convs ===================
        self.conv_downsample_xnet = torch.nn.Conv2d(in_channels=num_spectral, out_channels=num_spectral, kernel_size=KERNEL_SIZE, stride=factor,
                                               padding=KERNEL_SIZE // 2)
        self.conv_upsample_xnet = torch.nn.ConvTranspose2d(in_channels=num_spectral, out_channels=num_spectral, kernel_size=KERNEL_SIZE, stride=factor,
                                                      padding=KERNEL_SIZE // 2)

        self.conv_topan_xnet = torch.nn.Conv2d(in_channels=num_spectral, out_channels=1, kernel_size=3, stride=1, padding=3 // 2)
        self.conv_tolms_xnet = torch.nn.Conv2d(in_channels=1, out_channels=num_spectral, kernel_size=3, stride=1, padding=3 // 2)

        # =================== spatial net ===================   
        self.spatial = Unet(num_spectral)

    def recon(self, features, lms, ms, pan, x_net):  # z = pan, y2 = ms
        DELTA = self.delta_0
        ETA = self.eta_0
        # LAMBDA1 = self.lambda1
        # LAMBDA2 = self.lambda2

        # y1: ms upsampled
        # y2: ms
        # z: pan
        sz = lms.shape

        # =================== recon term from lms and pan ===================
        down = self.conv_downsample(lms)
        err_lms = self.conv_upsample(down - ms, output_size=sz)

        to_pan = self.conv_topan(lms)
        err_pan = pan - to_pan
        err_pan = self.conv_tolms(err_pan) 
        
        # y1 = y1 - DELTA * (LAMBDA1 * err_lms + LAMBDA2 * err_pan) + DELTA * ETA * (features - y1)
        state1 = (
            (1 - DELTA * ETA) * lms + 
            DELTA * err_lms + 
            DELTA * err_pan + 
            DELTA * ETA * features
        )
        
        # average state1 and x_net
        x_net = (state1 + x_net) / 2
        
        # =================== Prior from x_net ===================
        
        down_xnet = self.conv_downsample_xnet(x_net)
        err_xnet_lms = self.conv_upsample_xnet(down_xnet - ms, output_size=sz)
        
        to_pan_xnet = self.conv_topan_xnet(x_net)
        err_xnet_pan = pan - to_pan_xnet
        err_xnet_pan = self.conv_tolms_xnet(err_xnet_pan)
        
        out = (
            (1 - DELTA * ETA) * x_net + 
            DELTA * err_xnet_lms + 
            DELTA * err_xnet_pan + 
            DELTA * ETA * features
        )
        
        # ========================================================
        
        return out
    
    def forward_unfolding_step(self, y1, y2, z, x_net):
        # z = pan, y1 = lms, y2 = ms
        # output = y1
        # prox: refinement, proxnet, v=conv_out

        # unet(lms) + residual
        output = self.spatial(y1)[0] + y1
        
        y1 = self.recon(output, y1, y2, z)
        conv_out = self.spatial(y1)[0] + y1
        return conv_out
    
    def forward_once(self, x, y1, y2, z, x_net):
        y1 = self.recon(x, y1, y2, z)
        conv_out = self.spatial(y1)[0] + x_net
        # conv_out = conv_out + y1

        return conv_out #+ y1
    
    def forward(self, x, y1, y2, z, x_net):
        # x: deq state
        
        y1 = self.recon(x, y1, y2, z, x_net)
        conv_out = self.spatial(y1)[0]
        
        return conv_out
    

# ==================================================================================
# X_net
# 

# UnfoldingDEQ: 
#   1. unet(lms)  ->   init x_now
#   2. deq for-loop  ->    update x_now
#       2.1 recon(feature, y1, y2, z)
#       2.2 unet(feature - x_net)


#* ==================================================================================
#* Huggingface pretrained model, used for pushing to hub
#* transformers package can be used to load the model from the hub

# class Unfolding_DEQ_Config(PretrainedConfig):
#     model_type = 'Unfolding_DEQ'
    
#     def __init__(self, 
#                  args,
#                  num_spectral,
#                  num_channel, 
#                  factor, 
#                  lambda1, 
#                  lambda2,
#                  *,
#                  if_deq: bool=False,):
#         super().__init__()
#         self.args = args
#         self.num_spectral = num_spectral
#         self.num_channel = num_channel
#         self.factor = factor
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#         self.if_deq = if_deq
        
# class Unfolding_DEQ(PreTrainedModel):
#     config_class = Unfolding_DEQ_Config
    
#     def __init__(self, model_cfg):
#         super().__init__(model_cfg)
#         # PreTrainedModel.__init__(self)
#         # cfg
#         num_spectral = model_cfg.num_spectral
#         num_channel = model_cfg.num_channel
#         factor = model_cfg.factor
#         lambda1 = model_cfg.lambda1
#         lambda2 = model_cfg.lambda2
#         if_deq = model_cfg.if_deq
#         self.args = model_cfg.args

#         # model
#         self.full_stage = Unfolding(num_spectral, num_channel, factor, lambda1, lambda2)
#         # self.spatial = Unet(num_spectral)
#         # self.spatial = nn.Conv2d(num_spectral, num_spectral, 1)
        
#         # deq
#         self.if_deq = if_deq
#         if if_deq:
#             self.deq = get_deq(self.args)
#             apply_norm(self.full_stage)
#             logger.info('Apply normalization to unfolding model.')
#         else:
#             logger.info('Do not use DEQ and unfold once.')

#     def forward(self, y1, y2, z, x_net):  # z = pan, y1 = lms, y2 = ms   
#         # apply deq conv weights' normalization
#         if self.if_deq:
#             reset_norm(self.full_stage)

#         # x_now = self.spatial(y1)[0] + y1
#         x_now = y1
        
#         if self.if_deq:
#             # unfolding closure
#             def Unfolding_func(x_now):
#                 return self.full_stage(x_now, y1, y2, z, x_net)
#             x_out, info = self.deq(Unfolding_func, x_now, solver_kwargs={'tau':self.args.tau})
            
#             if self.training:
#                 return x_out
#             else:
#                 x_out = x_out[-1]
#                 return x_out
#         else:
#             x_out = self.full_stage(x_now, y1, y2, z, x_net)
#             return x_out
        
    
# AutoConfig.register('Unfolding_DEQ', Unfolding_DEQ_Config)
# AutoModel.register(Unfolding_DEQ_Config, Unfolding_DEQ)

#* ==================================================================================
# Pytorch model

from model.base_model import BaseModel

class Unfolding_DEQ(BaseModel):    
    def __init__(self, 
                 num_spectral,
                 num_channel, 
                 factor, 
                 lambda1, 
                 lambda2,
                 if_deq,
                 patch_merge=True,
                 norm_apply=False,
                 args=None):
        super().__init__()
        # cfg
        self.num_spectral = num_spectral
        self.num_channel = num_channel
        self.factor = factor
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.if_deq = if_deq
        self.args = args
        self.norm_apply = norm_apply
        self.patch_merge = patch_merge

        # model
        self.full_stage = Unfolding(num_spectral, num_channel, factor, lambda1, lambda2)

        # x_net network
        self.x_net_gen = FusionNet(num_spectral, num_channel)
        state_dict = torch.load("/Data3/cao/ZiHanCao/exps/panformer/utils/ckpts/FusionNet.pth.tar")
        self.x_net_gen.load_state_dict(state_dict['state_dict'])
        print('loading FusionNet weights done')
        # set to none grad and not to update the parameters
        self.x_net_gen.eval()
        self.x_net_gen.zero_grad(set_to_none=True)
        self.x_net_gen.requires_grad_(False)
        
        # deq
        self.if_deq = if_deq
        if if_deq:
            assert args is not None, 'args is required for DEQ'
            
            self.deq = get_deq(self.args)
            if norm_apply:
                apply_norm(self.full_stage)
                logger.info('Apply normalization to unfolding model, be careful of using EMA')
        else:
            logger.info('Do not use DEQ and unfold once.')

    def _forward_implem(self, y1, y2, z, x_net):  # z = pan, y1 = lms, y2 = ms   
        # apply deq conv weights' normalization
        if self.if_deq and self.norm_apply:
            reset_norm(self.full_stage)
        
        #* init vers
        # ver1: y1
        # x_now = y1

        # ver2: zeros init
        # x_now = torch.zeros_like(y1)

        # ver3: unet - spatial
        # reuse full_stage spatial net to init the x_now
        spatial_net = self.full_stage.spatial
        x_now = spatial_net(y1)[0] #+ y1
        
        if self.if_deq:
            # unfolding closure
            def Unfolding_func(x_now):
                return self.full_stage(x_now, y1, y2, z, x_net)
            x_out, info = self.deq(Unfolding_func, x_now,  solver_kwargs={'tau':self.args.tau})

            if self.training:
                return x_out
            else:
                x_out = x_out[-1]
                return x_out
        else:
            x_now = torch.zeros_like(y1)
            x_out = self.full_stage(x_now, 0, y1, y2, z, x_net)
            return x_out
        
    def _construct_ms(self, lms):
        de_factor = 1 / self.factor
        ms_size = (int(lms.size(2) * de_factor), int(lms.size(3) * de_factor))
        ms = torch.nn.functional.interpolate(lms, ms_size, mode='bilinear')
        return ms
        
    def sharpening_train_step(self, lms, pan, gt, criterion, **kwargs):
        if x_net := kwargs.get('x_net', None) is None:
            with torch.no_grad():
                x_net = self.x_net_gen(lms, pan).detach() + lms

        # x_net is set to None
        ms = self._construct_ms(lms)
        sr_s = self._forward_implem(lms, ms, pan, x_net)
        sr = sr_s[0]
        for idx in range(len(sr_s)):
            sr_s[idx] = sr_s[idx].clip(0, 1)

        loss = criterion(sr, gt)
        return sr, loss
    
    @torch.no_grad()
    def sharpening_val_step(self, lms, pan, **kwargs):
        ms = self._construct_ms(lms) if kwargs.get('ms', None) is None else kwargs['ms']
        if x_net := kwargs.get('x_net', None) is None:
            # x_net = torch.zeros_like(lms)
            with torch.no_grad():
                x_net = self.x_net_gen(lms, pan).detach() + lms

        if self.patch_merge:
            from model.module import PatchMergeModule
            logger.debug(f"using patch merge module")
            
            _patch_merge_model = PatchMergeModule(
                self,
                crop_batch_size=64,
                patch_size_list=[16 * self.factor, 16, 16 * self.factor],
                scale=1,
                patch_merge_step=self.patch_merge_step,
            )
            sr = _patch_merge_model.forward_chop(lms, ms, pan)[0]
        else:
            sr = self._forward_implem(lms, ms, pan, x_net)

        return sr.clip(0, 1)
    
    def patch_merge_step(self, lms, ms, pan, **kwargs):
        if x_net := kwargs.get('x_net', None) is None:
            x_net = torch.zeros_like(lms)
            with torch.no_grad():
                x_net = self.x_net_gen(lms, pan) + lms
            
        sr_s = self._forward_implem(lms, ms, pan, x_net)
        if isinstance(sr_s, (list, tuple)):
            sr_s = sr_s[0]
        elif isinstance(sr_s, torch.Tensor):
            pass
        else:
            raise ValueError('sr_s must be Tensor or list of Tensors')
        
        return sr_s
    
# ==================================================================================
        
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):   ## initialization for Conv2d

                variance_scaling_initializer(m.weight)  # method 1: initialization
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x/10*1.28

    variance_scaling(tensor)

    return tensor

if __name__ == '__main__':
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--tau', default=0.5, type=float)
    args = args_parser.parse_args()

    lms = torch.randn(1, 8, 64, 64).cuda()
    pan = torch.randn(1, 1, 64, 64).cuda()
    ms = torch.randn(1, 8, 16, 16).cuda()
    
    # cfg    
    # cfg = AutoConfig.for_model('Unfolding_DEQ', 
    #                                 args=args, 
    #                                 num_spectral=8, 
    #                                 num_channel=32, 
    #                                 factor=4, 
    #                                 lambda1=0.1, 
    #                                 lambda2=0.1, 
    #                                 if_deq=False)
    
    # model = AutoModel.from_config(cfg).cuda()
    
    
    model = Unfolding_DEQ(num_spectral=8,
                           num_channel=32,
                           factor=4,
                           lambda1=0.1,
                           lambda2=0.1,
                           if_deq=True,
                           args=args).cuda()
    
    # test EMA
    # from utils import EMA
    
    # ema_model = EMA(model)
    # ema_model.update()
    # print(ema_model.ema_model.state_dict())
    
    
    # get real sample
    import h5py
    file = h5py.File('/Data3/cao/ZiHanCao/datasets/pansharpening/wv3/reduced_examples/test_wv3_multiExm1.h5', 'r')
    lms = file['lms'][0:1] / 2047
    pan = file['pan'][0:1] / 2047
    gt = file['gt'][0:1] / 2047
    
    lms = torch.from_numpy(lms).float().cuda()
    pan = torch.from_numpy(pan).float().cuda()
    gt = torch.from_numpy(gt).float().cuda()
    
    print(lms.shape, pan.shape, gt.shape)
    
    sr = model.sharpening_train_step(lms, pan, gt, nn.L1Loss())

    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         if p.grad is None:
    #             print(f'{name} has not grad')
    #         else:
    #             print(f'{name} has grad shape as {p.grad.shape}')

    # test optimizer
    
    # from heavyball import PrecondSchedulePaLMSOAP
    
    # optimizer = PrecondSchedulePaLMSOAP(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # # train step
    # for i in range(10):
    #     x_net = torch.randn(1, 8, 64, 64).cuda()
    #     sr, loss = model.sharpening_train_step(lms, pan, lms, nn.L1Loss())

    #     optimizer.zero_grad()
    #     loss.backward()
    #     print(f'step {i}:', sr.shape, loss)
    #     optimizer.step()
        