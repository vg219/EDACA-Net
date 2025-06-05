from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float
    with_ckpt: bool

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int, with_ckpt=False):
        super().__init__()
        self.in_channels = in_channels
        self.with_ckpt = with_ckpt

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward_implem(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))
    
    def forward(self, x: Tensor) -> Tensor:
        if self.with_ckpt:
            return checkpoint(self.forward_implem, x, use_reentrant=False)
        else:
            return self.forward_implem(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, with_ckpt=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.with_ckpt = with_ckpt

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward_implem(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h
    
    def forward(self, x):
        if self.with_ckpt:
            return checkpoint(self.forward_implem, x, use_reentrant=False)
        else:
            return self.forward_implem(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        with_checkpoint=False,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, with_ckpt=with_checkpoint))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, with_ckpt=with_checkpoint)
        self.mid.attn_1 = AttnBlock(block_in, with_ckpt=with_checkpoint)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, with_ckpt=with_checkpoint)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
        with_checkpoint=False,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, with_ckpt=with_checkpoint)
        self.mid.attn_1 = AttnBlock(block_in, with_ckpt=with_checkpoint)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, with_ckpt=with_checkpoint)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, with_ckpt=with_checkpoint))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class LikelihoodVAE(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            with_ckpt=params.with_ckpt,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            with_ckpt=params.with_ckpt,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
    
    

if __name__ == '__main__':
    import numpy as np
    import PIL.Image as Image
    import sys
    sys.path.append('./')
    from model.lfq_vae.lfq import LFQ
    import accelerate
    from accelerate import Accelerator
    import torch
    torch.autograd.set_detect_anomaly(True)
    
    accelerator = Accelerator()
    
    # weight_d = torch.load('/Data3/cao/ZiHanCao/exps/panformer/model/lfq_vae/ckpts/imagenet_256_L.ckpt', weights_only=True)['state_dict']
    # enc_d = {}
    # dec_d = {}
    # for k, v in weight_d.items():
    #     if k.startswith('encoder'):
    #         enc_d[k.replace('encoder.', '')] = v
    #     elif k.startswith('decoder'):
    #         dec_d[k.replace('decoder.', '')] = v
    
    lfq = LFQ(dim=18, channel_first=True).to(accelerator.device)
    
    encoder = Encoder(ch=128, z_channels=9, in_channels=3, ch_mult=(1, 1, 2, 2), resolution=128, num_res_blocks=4, with_ckpt=True).to(accelerator.device)
    decoder = Decoder(ch=128, out_ch=3, z_channels=18, in_channels=3, ch_mult=(1, 1, 2, 2), resolution=128, num_res_blocks=4, with_ckpt=True).to(accelerator.device)
        
    # # compile
    print('compiling encoder and decoder')
    encoder = torch.compile(encoder)
    decoder = torch.compile(decoder)
        
    # Loss function
    from model.lfq_vae.gan_loss.loss import VQLPIPSWithDiscriminator
    loss_fn = VQLPIPSWithDiscriminator(disc_start=0, codebook_weight=0.1, pixelloss_weight=1.0, disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=0.8, commit_weight=0.25, codebook_enlarge_ratio=0, codebook_enlarge_steps=2000, perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, disc_ndf=64, disc_loss="non_saturate", gen_loss_weight=0.1, lecam_loss_weight=0.01).to(accelerator.device)
    loss_fn.discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(loss_fn.discriminator)
    
    # optimizer
    gen_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(lfq.parameters()), lr=1e-4)
    disc_optimizer = torch.optim.Adam(loss_fn.discriminator.parameters(), lr=1e-4)
    
    # encoder.load_state_dict(enc_d)
    # decoder.load_state_dict(dec_d)
    
    # Prepare models for DDP
    encoder, decoder, lfq, loss_fn.discriminator = accelerator.prepare(encoder, decoder, lfq, loss_fn.discriminator)
    gen_optimizer, disc_optimizer = accelerator.prepare(gen_optimizer, disc_optimizer)
    
    if accelerator.is_main_process:
        print('encoder and decoder loaded')
    
    # Load and preprocess image only on main process
    if accelerator.is_main_process:
        img = Image.open('/Data3/cao/ZiHanCao/exps/panformer/visualized_img/RWKVFusion_v11_RWKVFusion/mff_whu_v3/68.jpg').convert('RGB')
        img = np.array(img.resize((256, 256)))
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.
    else:
        img = torch.zeros(1, 3, 256, 256)
        
    img = img.to(accelerator.device)
    img_in = img.clone() * 2 - 1
    
    # Forward pass
    with accelerator.autocast():
        encoded = encoder(img)
        if accelerator.is_main_process:
            print('encoded: ', encoded.shape, 'requires_grad', encoded.requires_grad)
        
        (quantized, indices, aux_loss), loss_breakdown = lfq(encoded, inv_temperature=100., return_loss_breakdown=True)
        if accelerator.is_main_process:
            print('quantized: ', quantized.shape, 'requires_grad', quantized.requires_grad)
            print('indices: ', indices.shape, 'requires_grad', indices.requires_grad) 
            print('aux_loss: ', aux_loss, 'requires_grad', aux_loss.requires_grad)
        
        decoded = decoder(quantized)
        if accelerator.is_main_process:
            print('decoded: ', decoded.shape, 'requires_grad', decoded.requires_grad)
            
        def get_last_layer_weight():
            return accelerator.unwrap_model(decoder).conv_out.weight

        # Backward pass
        # encoder.requires_grad_(True)
        # decoder.requires_grad_(True)
        # lfq.requires_grad_(True)
        # loss_fn.discriminator.requires_grad_(False)
        gen_optimizer.zero_grad()
        vq_loss, log = loss_fn(aux_loss, loss_breakdown, img, decoded, 0, 0, get_last_layer_weight())
        accelerator.backward(vq_loss)
        gen_optimizer.step()
        
        
        # Print gradients only on main process
        if accelerator.is_main_process:
            for name, p in encoder.named_parameters():
                if p.grad is None:
                    print('encoder ', name, 'has no grad')
                    
            print('-' * 100)
            
            for name, p in decoder.named_parameters():
                if p.grad is None:
                    print('decoder ', name, 'has no grad')
                    
            print('-' * 100)
            
        accelerator.wait_for_everyone()
        
        # encoder.requires_grad_(False)
        # decoder.requires_grad_(False)
        # lfq.requires_grad_(False)
        # loss_fn.discriminator.requires_grad_(True)
        disc_optimizer.zero_grad()
        disc_loss, log = loss_fn(aux_loss, loss_breakdown, img.detach(), decoded.detach(), 1, 0, get_last_layer_weight())
        accelerator.backward(disc_loss)
        disc_optimizer.step()
        
        if accelerator.is_main_process:
            print('vq_loss: ', vq_loss, 'requires_grad', vq_loss.requires_grad)
            print('disc_loss: ', disc_loss, 'requires_grad', disc_loss.requires_grad)
            
            loss_fn_discriminator = loss_fn.discriminator
            for name, p in loss_fn_discriminator.named_parameters():
                if p.grad is None:
                    print('discriminator ', name, 'has no grad')
            
            print('-' * 100)
    
    accelerator.wait_for_everyone()
    
    