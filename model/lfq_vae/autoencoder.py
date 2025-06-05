import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def swish(x):
    # swish
    return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        use_conv_shortcut=False,
        use_agn=False,
        with_checkpoint=False,
    ) -> None:
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn
        self.with_checkpoint = with_checkpoint
        if not use_agn:  ## agn is GroupNorm likewise skip it if has agn before
            self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)

        self.conv1 = nn.Conv2d(
            in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False
        )

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_filters, out_filters, kernel_size=(1, 1), padding=0, bias=False
                )

    def forward_implem(self, x, **kwargs):
        residual = x

        if not self.use_agn:
            x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual
    
    def forward(self, x, **kwargs):
        if self.with_checkpoint:
            return checkpoint(self.forward_implem, x, use_reentrant=False)
        else:
            return self.forward_implem(x)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        in_channels,
        num_res_blocks,
        z_channels,
        ch_mult=(1, 2, 2, 4),
        resolution=128,
        with_checkpoint=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.z_channels = z_channels
        self.resolution = resolution

        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)

        self.conv_in = nn.Conv2d(
            in_channels, ch, kernel_size=(3, 3), padding=1, bias=False
        )

        ## construct the model
        self.down = nn.ModuleList()

        in_ch_mult = (1,) + tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]  # [1, 1, 2, 2, 4]
            block_out = ch * ch_mult[i_level]  # [1, 2, 2, 4]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out, with_checkpoint=with_checkpoint))
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(
                    block_out, block_out, kernel_size=(3, 3), stride=(2, 2), padding=1
                )

            self.down.append(down)

        ### mid
        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in, with_checkpoint=with_checkpoint))

        ### end
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, z_channels, kernel_size=(1, 1))

    def forward_implem(self, x):

        ## down
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)

            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)

        ## mid
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x
    
    def forward(self, x):
        return self.forward_implem(x)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        in_channels,
        num_res_blocks,
        z_channels,
        ch_mult=(1, 2, 2, 4),
        resolution=128,
        with_checkpoint=False,
    ) -> None:
        super().__init__()

        self.ch = ch
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        block_in = ch * ch_mult[self.num_blocks - 1]

        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=(3, 3), padding=1, bias=True
        )

        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in, with_checkpoint=with_checkpoint))

        self.up = nn.ModuleList()

        self.adaptive = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_in))
            for i_block in range(self.num_res_blocks):
                # if i_block == 0:
                #     block.append(ResBlock(block_in, block_out, use_agn=True))
                # else:
                block.append(ResBlock(block_in, block_out, with_checkpoint=with_checkpoint))
                block_in = block_out

            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)

        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=(3, 3), padding=1)

    def forward_implem(self, z):
        style = z.clone()  # for adaptive groupnorm
        
        z = self.conv_in(z)

        ## mid
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)

        ## upsample
        for i_level in reversed(range(self.num_blocks)):
            ### pass in each resblock first adaGN
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)

            if i_level > 0:
                z = self.up[i_level].upsample(z)

        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)

        return z
    
    def forward(self, z):
        return self.forward_implem(z)


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Depth-to-Space DCR mode (depth-column-row) core implementation.

    Args:
        x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
        block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)

    return x


class Upsampler(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(
            num_groups=32, num_channels=in_filters, eps=eps, affine=False
        )
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps

    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        # quantizer = F.adaptive_avg_pool2d(quantizer, (1, 1))
        ### calcuate var for scale
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps  # not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        ### calculate mean for bias
        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)

        x = self.gn(x)
        x = scale * x + bias

        return x


if __name__ == '__main__':
    import numpy as np
    import PIL.Image as Image
    import sys
    sys.path.append('./')
    from model.lfq_vae.quantizer import LFQ
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
    
    encoder = Encoder(ch=128, z_channels=18, in_channels=3, ch_mult=(1, 1, 2, 2), resolution=128, num_res_blocks=4, with_checkpoint=True).to(accelerator.device)
    decoder = Decoder(ch=128, out_ch=3, z_channels=18, in_channels=3, ch_mult=(1, 1, 2, 2), resolution=128, num_res_blocks=4, with_checkpoint=True).to(accelerator.device)
        
    # compile
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
    

    
    # import matplotlib.pyplot as plt
    
    # # raw image and reconstructed image
    # img = img[0].permute(1, 2, 0).cpu().numpy()
    # decoded = decoded * 0.5 + 0.5
    # decoded = decoded[0].detach().permute(1, 2, 0).cpu().numpy()
    
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.title('Original Image')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(decoded)
    # plt.title('Reconstructed Image')
    
    # plt.savefig('reconstructed_image.png')
    
    