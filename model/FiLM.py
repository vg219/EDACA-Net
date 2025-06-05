# taken from https://github.com/Zhaozixiang1228/IF-FILM/blob/main/net/Film.py

import torch
import torch.nn as nn
import torch.nn.functional as F


#################### Restormer Block ########################

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import numbers

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)
        # 这里可以在考虑一下结构，我觉得用MLP效果可能更好一点
        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  # 轻量化的时候这里设置成1
                 qkv_bias=False,flag=None):
        super(BaseFeatureExtraction, self).__init__()

        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
        self.reduce_channel = None
        if flag=='cat':
            self.reduce_channel = self.reduce_channel=nn.Conv2d(128, int(dim), kernel_size=1, bias=False).cuda()
    def forward(self, x):            
        if self.reduce_channel is not None:
            x= self.reduce_channel(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self,flag=None):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        # self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
        #                             stride=1, padding=0, bias=True)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)  
        if flag=='cat':
            self.shffleconv =nn.Conv2d(128, 64, kernel_size=1, bias=True)                                                   
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3,flag=None):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        if flag=="cat":
            INNmodules = [DetailNode(flag="cat") for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

# =============================================================================
# 开始最前面的Encoder
# =============================================================================
import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(RestormerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
    
class  RestormerFLow(nn.Module):
    def __init__(self,
                 inp_channels=2,
                 out_channels=1,
                 dim=32,
                 num_blocks=4,
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',act="Sigmoid"
                 ):

        super(RestormerFLow, self).__init__()

        self.patch_embed = nn.Conv2d(inp_channels,dim, kernel_size=3,stride=1, padding=1, bias=bias)

        self.encoder_level1 = nn.Sequential(*[RestormerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        # self.output = nn.Sequential(
        #     nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
        #               stride=1, padding=1, bias=bias),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
        #               stride=1, padding=1, bias=bias),)
        self.output = nn.Sequential(
            nn.Conv2d(int(dim),  out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)


        self.flag=0
        if act == "Sigmoid":
            self.act = nn.Sigmoid()              
        elif act == "ReLU":    
            self.flag=1
            self.act = nn.ReLU() 
        elif  act == "Softmax":  
            self.act = nn.Softmax(dim=1)     
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder_level1(x)
        x = self.output(x)

        x = self.act(x)
        if self.flag:
            x=(x+1e-7)/torch.sum((x+1e-7),dim=1,keepdim=True)

        #print(torch.max(x),torch.min(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)

        return attn_output


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x.permute(0, 2, 1))).permute(0, 2, 1)
        max_out = self.fc(self.max_pool(x.permute(0, 2, 1))).permute(0, 2, 1)
        out = avg_out + max_out
        # out = avg_out
        return self.sigmoid(out)


class imagefeature2textfeature(nn.Module):
    def __init__(self, in_channel, mid_channel, hidden_dim):
        super(imagefeature2textfeature, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.conv(x)

        x = F.interpolate(x, [288, 384], mode='nearest')
        x = x.contiguous().view(x.size(0), x.size().numel() // x.size(0) // self.hidden_dim, self.hidden_dim)
        return x


class restormer_cablock(nn.Module):
    def __init__(
            self,
            input_channel=1,
            restormerdim=32,
            restormerhead=8,
            image2text_dim=10,
            ffn_expansion_factor=4,
            bias=False,
            LayerNorm_type='WithBias',
            hidden_dim=768,
            pooling='avg',
            normalization='l1'
    ):
        super().__init__()
        self.convA1 = nn.Conv2d(input_channel, restormerdim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.preluA1 = nn.PReLU()
        self.convA2 = nn.Conv2d(image2text_dim, restormerdim, kernel_size=1)
        self.preluA2 = nn.PReLU()
        self.convA3 = nn.Conv2d(2 * restormerdim, restormerdim, kernel_size=1)
        self.preluA3 = nn.PReLU()

        self.convB1 = nn.Conv2d(input_channel, restormerdim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.preluB1 = nn.PReLU()
        self.convB2 = nn.Conv2d(image2text_dim, restormerdim, kernel_size=1)
        self.preluB2 = nn.PReLU()
        self.convB3 = nn.Conv2d(2 * restormerdim, restormerdim, kernel_size=1)
        self.preluB3 = nn.PReLU()

        self.image2text_dim = image2text_dim
        self.restormerA1 = RestormerBlock(restormerdim, restormerhead, ffn_expansion_factor, bias, LayerNorm_type)
        self.restormerB1 = RestormerBlock(restormerdim, restormerhead, ffn_expansion_factor, bias, LayerNorm_type)
        self.cross_attentionA1 = CrossAttention(embed_dim=hidden_dim, num_heads=8)
        self.cross_attentionA2 = CrossAttention(embed_dim=hidden_dim, num_heads=8)
        self.imagef2textfA1 = imagefeature2textfeature(restormerdim, image2text_dim, hidden_dim)
        self.imagef2textfB1 = imagefeature2textfeature(restormerdim, image2text_dim, hidden_dim)
        self.image2text_dim = image2text_dim



    def forward(self, imageA, imageB, text):
        if len(imageA.shape) == 3:
            imageA = imageA.cuda().unsqueeze(0).permute(0, 3, 1, 2)
            imageB = imageB.cuda().unsqueeze(0).permute(0, 3, 1, 2)
        b, _, H, W = imageA.shape

        imageA = self.restormerA1(self.preluA1(self.convA1(imageA)))
        imageAtotext = self.imagef2textfA1(imageA)
        imageB = self.restormerB1(self.preluB1(self.convB1(imageB)))
        imageBtotext = self.imagef2textfB1(imageB)

        ca_A = self.cross_attentionA1(text, imageAtotext, imageAtotext)
        imageA_sideout = imageA
        ca_A = torch.nn.functional.adaptive_avg_pool1d(ca_A.permute(0, 2, 1), 1).permute(0, 2, 1)
        ca_A = F.normalize(ca_A, p=1, dim=2)

        ca_A = (imageAtotext * ca_A).view(imageA.shape[0], self.image2text_dim, 288, 384)
        imageA_sideout = F.interpolate(imageA_sideout, [H, W], mode='nearest')
        ca_A = F.interpolate(ca_A, [H, W], mode='nearest')
        ca_A = self.preluA3(
            self.convA3(torch.cat(
                (F.interpolate(imageA, [H, W], mode='nearest'), self.preluA2(self.convA2(ca_A)) + imageA_sideout), 1)))

        ca_B = self.cross_attentionA2(text, imageBtotext, imageBtotext)
        imageB_sideout = imageB
        ca_B = torch.nn.functional.adaptive_avg_pool1d(ca_B.permute(0, 2, 1), 1).permute(0, 2, 1)
        ca_B = F.normalize(ca_B, p=1, dim=2)

        ca_B = (imageBtotext * ca_B).view(imageA.shape[0], self.image2text_dim, 288, 384)
        imageB_sideout = F.interpolate(imageB_sideout, [H, W], mode='nearest')
        ca_B = F.interpolate(ca_B, [H, W], mode='nearest')
        ca_B = self.preluB3(
            self.convB3(torch.cat(
                (F.interpolate(imageB, [H, W], mode='nearest'), self.preluB2(self.convB2(ca_B)) + imageB_sideout), 1)))

        return ca_A, ca_B


class text_preprocess(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(text_preprocess, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class Net(nn.Module):
    def __init__(
            self,
            mid_channel=32,
            decoder_num_heads=8,
            ffn_factor=4,
            bias=False,
            LayerNorm_type='WithBias',
            out_channel=1,
            hidden_dim=256,
            image2text_dim=32,
            pooling='avg',
            normalization='l1'
    ):
        super().__init__()
        self.text_process = text_preprocess(768, hidden_dim)
        self.restormerca1 = restormer_cablock(hidden_dim=hidden_dim, image2text_dim=image2text_dim)
        self.restormerca2 = restormer_cablock(input_channel=mid_channel, hidden_dim=hidden_dim,
                                              image2text_dim=image2text_dim)
        self.restormerca3 = restormer_cablock(input_channel=mid_channel, hidden_dim=hidden_dim,
                                              image2text_dim=image2text_dim)
        self.restormer1 = RestormerBlock(2 * mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.restormer2 = RestormerBlock(mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.restormer3 = RestormerBlock(mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.conv1 = nn.Conv2d(2 * mid_channel, mid_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=1)
        self.softmax = nn.Sigmoid()

    def forward(self, imageA, imageB, text):

        text = self.text_process(text)
        featureA, featureB = self.restormerca1(imageA, imageB, text)
        featureA, featureB = self.restormerca2(featureA, featureB, text)
        featureA, featureB = self.restormerca3(featureA, featureB, text)
        fusionfeature = torch.cat((featureA, featureB), 1)
        fusionfeature = self.restormer1(fusionfeature)
        fusionfeature = self.conv1(fusionfeature)
        fusionfeature = self.restormer2(fusionfeature)
        fusionfeature = self.restormer3(fusionfeature)
        fusionfeature = self.conv2(fusionfeature)
        fusionfeature = self.softmax(fusionfeature)
        return fusionfeature
    
    
if __name__ == '__main__':
    torch.cuda.set_device(0)
    
    model = Net().cuda()
    imageA = torch.randn(2, 1, 288, 384).cuda()
    imageB = torch.randn(2, 1, 288, 384).cuda()
    text = torch.randn(2, 288, 768).cuda()
    # out = model(imageA, imageB, text)
    # print(out.shape)
    
    # out.mean().backward()
    
    # print(torch.cuda.memory_summary('cuda:0'))
    
    
    
    out = model(imageA, imageB, text)
    # sr = torch.randn(1, 8, img_size * scale, img_size * scale).to(device)
    # loss = F.mse_loss(out, sr)
    loss = out.sum()
    print(loss)
    loss.backward()
    
    # find unused params and big-normed gradient
    d_grads = {}
    n_params = 0
    for n, p in model.named_parameters():
        n_params += p.numel()
        if p.grad is None:
            print(n, "has no grad")
        else:
            p_sum = torch.abs(p.grad).sum().item()
            d_grads[n] = p_sum

    ## topk
    d_grads = dict(sorted(d_grads.items(), key=lambda item: item[1], reverse=True))
    for k, v in list(d_grads.items())[:20]:
        print(k, v)

    # ## params
    print("total params:", n_params / 1e6, "M")


    