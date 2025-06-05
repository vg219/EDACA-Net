from functools import partial
from typing import Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from kornia.losses.focal import FocalLoss

from accelerate import Accelerator

import sys
sys.path.append("./")
from model.downstream_module.base_downstream_model import register_seg_module, get_seg_module

logger = Console(log_path=False)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1.0):
        super(DiceLoss, self).__init__()
        # weight  # 类别权重，可以用于处理类别不平衡
        self.smooth = smooth
        
        self.register_buffer("weight", weight)

    def forward(self, inputs, targets):
        # 输入 inputs 的形状应为 [N, C, H, W]
        # targets 的形状应为 [N, H, W]，其中的值为类别索引（0 到 C-1）
        
        # 将 targets 转换为 one-hot 编码
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # 使用 softmax 函数来限制 inputs 的输出范围在 [0, 1] 之间
        inputs = F.softmax(inputs, dim=1)
        
        # 计算每个类别的交集和并集
        intersection = torch.sum(inputs * targets, dim=(2, 3))
        cardinality = torch.sum(inputs + targets, dim=(2, 3))
        
        # 计算每个批次和每个类别的 Dice 系数
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # 如果有权重，则应用权重
        if self.weight is not None:
            dice_score = dice_score * self.weight
        
        # 计算平均 Dice 损失
        dice_loss = 1 - dice_score.mean(dim=1)  # 对所有类别求平均
        return dice_loss.mean()  # 对所有批次求平均
    

def to_two_tuple(x: Any):
    return (x, x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.add_res = (
            nn.Conv2d(in_channels, out_channels, 1)
            if out_channels != in_channels
            else nn.Identity()
        )

    def forward(self, x):
        feat = self.relu(self.bn(self.conv(x)))
        return feat + self.add_res(x)


@register_seg_module("conv_seg")
class FusionSegConvModule(nn.Module):
    def __init__(
        self,
        decoder_channels: list[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.decoder_channels = decoder_channels
        self.dropout = dropout

        in_feat = []
        for i in range(len(self.decoder_channels) - 1):
            in_feat.append(ConvBlock(self.decoder_channels[i], self.decoder_channels[i + 1]))
            if dropout != 0.:
                in_feat.append(nn.Dropout(dropout))

        self.decoder = nn.Sequential(*in_feat)

    def forward(self, x):
        return self.decoder(x)
    

def mask_ignore_index(pred: "torch.Tensor", gt: "torch.Tensor", ignore_index: int=0):
    if ignore_index is not None:
        mask = (gt != ignore_index).float()
        pred = pred * mask.unsqueeze(1)
        gt = gt * mask
            
    return pred, gt

def get_ignore_weight(n_class: int, ignore_index: int=0, ignore_weight: float=0.2):
    weight = torch.tensor([1 if i != ignore_index else ignore_weight for i in range(n_class)])
    # weight = weight.float()[None, :, None, None]
    
    return weight


class SegFocalLoss(nn.Module):
    def __init__(self, n_class: int=9, alpha=0.5, gamma: float=2.0):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        
    def forward(self, pred: "torch.Tensor", gt: "torch.Tensor"):
        gt = gt.long()
        if gt.ndim == 4:
            assert gt.size(1) == 1
            gt = gt[:, 0]
        
        # mask_ignore_index(pred, gt, self.ignore_index)
        
        # pred = pred * self.weight[None, :, None, None].to(device=pred.device)
        # if self.ignore_index is not None:
        #     gt[gt == self.ignore_index] = 0.
        
        return self.focal_loss(pred, gt).mean()


seg_loss = {
    'cross_entropy': nn.CrossEntropyLoss,
    'dice_loss': DiceLoss,
    'focal_loss': SegFocalLoss,
}


class FusionSegmentationPlugger(nn.Module):
    def __init__(
        self,
        decoder_chans: list[int],
        num_classes: int,
        basic_block: "str | list[str]",
        feature_injection_type: Literal["add", "cat", "per-layer"] = "per-layer",
        seg_module_cfg: "dict | list[dict]" = {},
        n_block_per_layer: "int | list[int]" = 1,
        feature_chans: "list[int] | int"=None,
        last_hidden_chan: int=None,
        output_seg_size: "int | tuple[int]"=None,
        reverse_feat: bool=True,
        loss_type: str="cross_entropy",
        loss_cfg: dict={},
        include_bg: bool=True,
    ):
        super().__init__()

        self.decoder_chans = decoder_chans
        self.num_classes = num_classes
        self.basic_block = basic_block
        self.feature_injection_type = feature_injection_type
        self.n_block_per_layer = n_block_per_layer
        self.feature_chans = feature_chans
        self.last_hidden_chan = last_hidden_chan
        self.output_seg_size = output_seg_size
        self.seg_module_cfg = seg_module_cfg
        self.reverse_feat = reverse_feat
        self.check_init()

        # feature conv in
        if self.feature_injection_type == "per-layer":
            self.feature_conv_in = nn.ModuleList([])
            for i in range(len(self.feature_chans)):
                self.feature_conv_in.append(
                    ConvBlock(self.feature_chans[i], self.decoder_chans[i])
                )
        else:
            assert isinstance(self.feature_chans, int)
            self.feature_conv_in = ConvBlock(self.feature_chans, self.decoder_chans[0])
        
        # main feature
        self.seg_module = nn.ModuleList([])
        if isinstance(basic_block, list):
            for i in range(len(basic_block)):
                basic_seg_cls = get_seg_module(basic_block[i])
                logger.log(f"seg head layer {i}: {basic_seg_cls.__name__}")
                
                basic_seg = [basic_seg_cls(**seg_module_cfg[i]) for _ in range(n_block_per_layer[i])]
                basic_seg = nn.Sequential(*basic_seg)
                
                self.seg_module.append(basic_seg)
        else:
            basic_seg_cls = get_seg_module(basic_block)
            logger.log(f"seg head: {basic_seg_cls.__name__}")
            
            basic_seg = [basic_seg_cls(**seg_module_cfg) for _ in range(n_block_per_layer)]
            basic_seg = nn.Sequential(*basic_seg)
            
            self.seg_module.append(basic_seg)
        
        self.to_num_classes = nn.Conv2d(last_hidden_chan, num_classes, 1)
        self.loss_critertion = seg_loss[loss_type](**loss_cfg)

    def check_init(self):
        assert self.feature_injection_type in ["add", "cat", "per-layer"]
        if isinstance(self.basic_block, list):
            self.n_layer = len(self.basic_block)
            logger.log(f"set {self.n_layer} layers of segmentation module")

            assert isinstance(
                self.seg_module_cfg, list
            ), "when basic_block is a list, seg_module_cfg should also be a list of dict"
            assert len(self.basic_block) == len(
                self.seg_module_cfg
            ), "when basic_block is a list, basic_block and seg_module_cfg should have the same length"
            assert len(self.n_block_per_layer) == len(
                self.basic_block
            ), "when basic_block is a list, n_block_per_layer should be a list of integers with the same length as basic_block"
            if self.feature_injection_type == "per-layer":
                assert len(self.basic_block) == len(
                    self.feature_chans
                ), "when feature_injection_type is per-layer, basic_block should be a list of strings with the same length as feature_sizes"
                
        if self.feature_injection_type == "add":
            self.inject_fn = lambda x, y: x + y
        elif self.feature_injection_type == "cat":
            self.inject_fn = lambda x, y: torch.cat([x, y], dim=1)
        else:
            self.inject_fn = None
            
        assert isinstance(self.output_seg_size, (int, tuple))
        if isinstance(self.output_seg_size, int):
            self.output_seg_size = to_two_tuple(self.output_seg_size)

    def interpolating(self, x, y):
        # interpolate y to x size
        # or a given size
        
        if self.output_seg_size is not None and self.training:
            size = self.output_seg_size
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        elif x.shape[-2:] != y.shape[-2:]:
            size = x.shape[-2:]
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        
        return x, y
    
    def list_feat_forward(self, features: "list[torch.Tensor]"):
        if self.reverse_feat:
            features = list(reversed(features))
        f = features[0]
        if self.feature_injection_type in ["add", "cat"]:
            for fi in range(1, len(features)):
                f, f_in = self.interpolating(f, f_in)
                f = self.inject_fn(f, f_in)

            f = self.feature_conv_in(f)
            f = self.seg_module[0](f)
            for i in range(1, len(self.seg_module)):
                f = self.seg_module[i](f)
        else:
            # assume we use the largest feats to be the output size of seg_map
            f = self.feature_conv_in[0](f)
            f = self.seg_module[0](f)
            for i in range(1, len(self.seg_module)):
                f_in = features[i]
                f_in = self.feature_conv_in[i](f_in)
                f, f_in = self.interpolating(f, f_in)
                f += f_in
                f = self.seg_module[i](f)
            
        return f

    def single_feat_forward(self, feature: "torch.Tensor"):
        f = feature
        
        for i in range(len(self.seg_module)):
            f = self.seg_module[i](f)
        return f
    
    def forward(self, features: "list[torch.Tensor] | torch.Tensor"):
        if torch.is_tensor(features):
            feat = self.single_feat_forward(features)
        else:
            feat = self.list_feat_forward(features)
            
        return self.to_num_classes(feat)
    
    def loss(self, pred: "torch.Tensor", gt: "torch.Tensor"):
        # TODO: support multi-scale segmentation maps
        
        # seg mask background to 255 ignore
        # bg_mask = gt == 0
        # gt[gt >= 1] = gt[gt >= 1] - 1
        # gt[bg_mask] = pred.shape[1] - 1
        # gt = gt.long()
        
        if gt.ndim == 4:
            assert gt.size(1) == 1
            gt = gt[:, 0]
            

        return self.loss_critertion(pred, gt) * 10
    
    
def seg_plugger_cfg_joint(feature_chans: "list[int]", 
                         output_size: "int | tuple[int]"):
    return dict(
        num_classes=2,  # ignore background mask (2)
        basic_block=["conv_seg"]*3,
        n_block_per_layer=[2, 2, 2],
        feature_injection_type="per-layer",
        seg_module_cfg=[
            dict(
                decoder_channels=[64, 64],
                dropout=0.,
            ),
            dict(
                decoder_channels=[64, 64],
                dropout=0.,
            ),
            dict(
                decoder_channels=[64, 64],
                dropout=0.,
            ),
        ],
        last_hidden_chan=64,
        output_seg_size=output_size,
        feature_chans=feature_chans, #[128, 64, 32],
        decoder_chans=[64, 64, 64],
        loss_type='cross_entropy',
        loss_cfg=dict(
            # n_class=9,
            # weight=get_ignore_weight(2, ignore_index=0, ignore_weight=1.0),
            # label_smoothing=0.1,
        ),
        reverse_feat=True
    )
    
    
if __name__ == "__main__":
    seg_plugger = FusionSegmentationPlugger(**seg_plugger_cfg_joint([3, 64, 128],
                                                                   output_size=64)).cuda()
    features = [
        torch.randn(1, 128, 16, 16).cuda(),
        torch.randn(1, 64, 32, 32).cuda(),
        torch.randn(1, 3, 64, 64).cuda(),
    ]
    
    seg_map = seg_plugger(features)
    print(seg_map.shape)
    loss = seg_plugger.loss(seg_map, torch.randint(0, 2, (1, 64, 64)).cuda())
    print(loss)
    
    loss.backward()
    
    for n, p in seg_plugger.named_parameters():
        if p.grad is None:
            print(n)
    
    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    
    # f = FlopCountAnalysis(seg_plugger, (features,))
    # print(flop_count_table(f))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    