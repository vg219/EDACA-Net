"""Model definition

In this file, we implement Stage-I Model (ConvVQModel), Lookup-free Quantization and Vector Quantization.
"""

import math
from typing import Mapping, Text, Tuple, Callable

import torch
import torch.nn.functional as F
from einops import rearrange, reduce

from modeling.modules import BaseModel
from modeling.quantizer.quantizer_utils import entropy_loss_fn


class Conv2dSame(torch.nn.Conv2d):
    """ Convolution wrapper for 2D convolutions using `SAME` padding."""
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """ Calculate padding such that the output has the same height/width when stride=1.

        Args:
            i -> int: Input size.
            k -> int: Kernel size.
            s -> int: Stride size.
            d -> int: Dilation rate.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the convolution applying explicit `same` padding.

        Args:
            x -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)


def GroupNorm(in_channels):
    """ GroupNorm with 32 groups."""
    if in_channels % 32 != 0:
        raise ValueError(f"GroupNorm requires in_channels to be divisible by 32, got {in_channels}.")
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class ResidualBlock(torch.nn.Module):
    """ Residual block with two convolutional layers."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        norm_func = GroupNorm
    ):
        """ Initializes the residual block.

        Args:
            in_channels -> int: Number of input channels.
            out_channels -> int: Number of output channels. Default is in_channels.
            norm_func -> Callable: Normalization function. Default is GroupNorm.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels is None else out_channels

        self.norm1 = norm_func(self.in_channels)
        self.conv1 = Conv2dSame(self.in_channels, self.out_channels, kernel_size=3, bias=False)

        self.norm2 = norm_func(self.out_channels)
        self.conv2 = Conv2dSame(self.out_channels, self.out_channels, kernel_size=3, bias=False)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv2dSame(self.out_channels, self.out_channels, kernel_size=1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the residual block.

        Args:
            hidden_states -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(hidden_states)

        return hidden_states + residual


class ResidualStage(torch.nn.Module):
    """ Residual stage with multiple residual blocks."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        norm_func = GroupNorm
    ):
        """ Initializes the residual stage.

        Args:
            in_channels -> int: Number of input channels.
            out_channels -> int: Number of output channels.
            num_res_blocks -> int: Number of residual blocks.
            norm_func -> Callable: Normalization function. Default is GroupNorm.
        """
        super().__init__()

        self.res_blocks = torch.nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(in_channels, out_channels, norm_func=norm_func))
            in_channels = out_channels

    def forward(self, hidden_states: torch.Tensor, *unused_args) -> torch.Tensor:
        """ Forward pass of the residual stage.

        Args:
            hidden_states -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for res_block in self.res_blocks:
            hidden_states = res_block(hidden_states)

        return hidden_states


class DownsamplingStage(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        sample_with_conv: bool = False,
        norm_func = GroupNorm
    ):
        """ Initializes the downsampling stage.

        Args:
            in_channels -> int: Number of input channels.
            out_channels -> int: Number of output channels.
            num_res_blocks -> int: Number of residual blocks.
            sample_with_conv -> bool: Whether to sample with a convolution or with a stride. Default is False.
            norm_func -> Callable: Normalization function. Default is GroupNorm.
        """
        super().__init__()

        self.res_blocks = torch.nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(in_channels, out_channels, norm_func))
            in_channels = out_channels

        self.sample_with_conv = sample_with_conv
        if self.sample_with_conv:
            self.down_conv = Conv2dSame(in_channels, in_channels, kernel_size=3, stride=2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the downsampling stage.

        Args:
            hidden_states -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for res_block in self.res_blocks:
            hidden_states = res_block(hidden_states)

        if self.sample_with_conv:
            hidden_states = self.down_conv(hidden_states)
        else:
            hidden_states = F.avg_pool2d(hidden_states, kernel_size=2, stride=2)

        return hidden_states


class UpsamplingStage(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        norm_func = GroupNorm
    ):
        """ Initializes the upsampling stage.

        Args:
            in_channels -> int: Number of input channels.
            out_channels -> int: Number of output channels.
            num_res_blocks -> int: Number of residual blocks.
            norm_func -> Callable: Normalization function. Default is GroupNorm.
        """
        super().__init__()

        self.res_blocks = torch.nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(in_channels, out_channels, norm_func))
            in_channels = out_channels

        self.upsample_conv = Conv2dSame(out_channels, out_channels, kernel_size=3)

    def forward(self, hidden_states: torch.Tensor, *unused_args) -> torch.Tensor:
        """ Forward pass of the upsampling stage.

        Args:
            hidden_states -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for res_block in self.res_blocks:
            hidden_states = res_block(hidden_states)

        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.upsample_conv(hidden_states)

        return hidden_states


class ConvEncoder(torch.nn.Module):
    def __init__(self, config):
        """ Initializes the convolutional encoder.

        Args:
            config: Configuration of the model architecture.
        """
        super().__init__()
        self.config = config

        self.conv_in = Conv2dSame(self.config.num_channels, self.config.hidden_channels, kernel_size=3, bias=False)

        in_channel_mult = (1,) + tuple(self.config.channel_mult)
        num_res_blocks = self.config.num_res_blocks
        hidden_channels = self.config.hidden_channels

        encoder_blocks = []
        for i_level in range(self.config.num_resolutions):
            in_channels = hidden_channels * in_channel_mult[i_level]
            out_channels = hidden_channels * in_channel_mult[i_level + 1]

            if i_level < (self.config.num_resolutions - 1):
                encoder_blocks.append(DownsamplingStage(in_channels, out_channels, num_res_blocks, self.config.sample_with_conv))
            else:
                encoder_blocks.append(ResidualStage(in_channels, out_channels, num_res_blocks))
        self.down = torch.nn.ModuleList(encoder_blocks)

        # middle
        mid_channels = out_channels
        self.mid = ResidualStage(mid_channels, mid_channels, num_res_blocks)

        # end
        self.norm_out = GroupNorm(mid_channels)
        self.conv_out = Conv2dSame(mid_channels, self.config.token_size, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the convolutional encoder.

        Args:
            pixel_values -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # downsampling
        hidden_states = self.conv_in(pixel_values)
        
        for block in self.down:
            hidden_states = block(hidden_states)
        # middle
        hidden_states = self.mid(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class ConvDecoder(torch.nn.Module):
    def __init__(self, config):
        """ Initializes the convolutional decoder.

        Args:
            config: Configuration of the model architecture.
        """
        super().__init__()

        self.config = config

        # compute in_channel_mult, block_in and curr_res at lowest res
        block_in = self.config.hidden_channels * self.config.channel_mult[self.config.num_resolutions - 1]
        num_res_blocks = self.config.get("num_res_blocks_decoder", self.config.num_res_blocks)
        hidden_channels = self.config.hidden_channels
        in_channel_mult = tuple(self.config.channel_mult) + (self.config.channel_mult[-1],)

        # z to block_in
        if config.quantizer_type == "vae":
            self.conv_in = Conv2dSame(self.config.token_size // 2, block_in, kernel_size=3)
        else:
            self.conv_in = Conv2dSame(self.config.token_size, block_in, kernel_size=3)

        # middle
        self.mid = ResidualStage(block_in, block_in, num_res_blocks)

        # upsampling
        decoder_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            in_channels = hidden_channels * in_channel_mult[i_level + 1]
            out_channels = hidden_channels * in_channel_mult[i_level]
            if i_level > 0:
                decoder_blocks.append(UpsamplingStage(in_channels, out_channels, num_res_blocks))
            else:
                decoder_blocks.append(ResidualStage(in_channels, out_channels, num_res_blocks))
        self.up = torch.nn.ModuleList(decoder_blocks)

        # end
        self.norm_out = GroupNorm(out_channels)
        self.conv_out = Conv2dSame(out_channels, self.config.num_channels, kernel_size=3)

    def forward(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the convolutional decoder.

        Args:
            z_quantized -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # z to block_in
        hidden_states = self.conv_in(z_quantized)

        # middle
        hidden_states = self.mid(hidden_states)

        # upsampling decoder
        for block in self.up:
            hidden_states = block(hidden_states, z_quantized)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class LookupFreeQuantizer(torch.nn.Module):
    def __init__(
        self,
        token_bits: int = 10,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.1,
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
    ):
        """ Initializes the lookup-free quantizer.

        Args:
            token_bits -> int: The number of bits per token.
            commitment_cost -> float: The commitment cost.
            entropy_loss_weight -> float: The weight of the entropy loss.
            entropy_loss_temperature -> float: The temperature for the entropy loss.
            entropy_gamma -> float: The gamma for the entropy loss.
        """
        super().__init__()
        self.token_size = token_bits
        self.codebook_size = 2 ** token_bits

        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma

        bits_to_indices = torch.pow(2.0, torch.arange(0, self.token_size, dtype=torch.float32))
        self.register_buffer('bits_to_indices', bits_to_indices.int())

        all_codes = torch.arange(self.codebook_size)
        bits = ((all_codes[..., None].int() & self.bits_to_indices) != 0).float()
        self.register_buffer('codebook', bits * 2.0 - 1.0)


    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Forward pass of the quantizer.

        Args:
            z -> torch.Tensor: The input tensor.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        ones = torch.ones_like(z)
        sign_mask = (z > 0.0)
        z_quantized = torch.where(sign_mask, ones, -ones)

        min_encoding_indices = self.convert_bits_to_indices(z_quantized)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        entropy_loss = torch.zeros((), device=z.device)
        per_sample_entropy = torch.zeros((), device=z.device)
        avg_entropy = torch.zeros((), device=z.device)

        # Use entropy loss on the codebook
        if self.entropy_loss_weight != 0.0 and self.training:
            d = - 2 * torch.einsum('b h w c, n c -> b h w n', z, self.codebook)

            per_sample_entropy, avg_entropy = entropy_loss_fn(-1*d, self.entropy_loss_temperature, self.entropy_gamma)
            entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)

        loss = commitment_loss + entropy_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            min_encoding_indices=min_encoding_indices
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """ Returns the `codebook entry` for the given indices.

        As the codebook exists only implicitly, this is mainly an integer conversion to a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.

        Returns:
            tokens -> torch.Tensor: The bit representation.
        """
        indices = indices.long()
        bits = ((indices[..., None].int() & self.bits_to_indices) != 0).float()
        tokens = bits * 2.0 - 1.0  # scale to -1..1
        return tokens

    def convert_bits_to_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        """ Converts the given tokens to index numbers.

        As the codebook exists only implicitly, this is mainly an integer conversion from a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            tokens -> torch.Tensor: The tokens.

        Returns:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.
        """
        tokens = rearrange(tokens, 'b h w c -> b h w c').contiguous()
        sign_mask = (tokens > 0.0)
        return reduce(sign_mask.int() * self.bits_to_indices, 'b h w c -> b h w', 'sum')

    def convert_indices_to_bits(self, indices: torch.Tensor) -> torch.Tensor:
        """ Converts the given indices to tokens.

        As the codebook exists only implicitly, this is mainly an integer conversion to a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.

        Returns:
            tokens -> torch.Tensor: The bit representation.
        """
        indices = indices.long()
        return self.get_codebook_entry(indices)



class SimpleVectorizer(torch.nn.Module):
    """
    Inspired by https://github.com/google-research/magvit/blob/main/videogvt/models/vqvae.py
    """
    def __init__(
        self,
        codebook_size: int = 1024,
        token_size: int = 256,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.0,
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
        use_l2_normalisation: bool = False,
    ):
        """ Initializes the quantizer.

        Args:
            codebook_size -> int: The size of the codebook.
            token_size -> int: The feature dimensions of the tokens.
            commitment_cost -> float: The commitment cost.
            entropy_loss_weight -> float: The weight of the entropy loss.
            entropy_loss_temperature -> float: The temperature of the entropy loss.
            entropy_gamma -> float: The gamma of the entropy loss.
            use_l2_normalisation -> bool: Whether to use L2 normalisation.
        """

        super().__init__()
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma
        self.use_l2_normalisation = use_l2_normalisation

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Computes the quantization loss and returns the quantized latent representation.

        Args:
            z -> torch.Tensor: The latent representation.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()

        if self.use_l2_normalisation:
            z = torch.nn.functional.normalize(z, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight

        z_flattened = rearrange(z, 'b h w c -> (b h w) c')

        # distances from z to embeddings e_j d = (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1)
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)
        entropy_loss = torch.zeros((), device=z.device)
        per_sample_entropy = torch.zeros((), device=z.device)
        avg_entropy = torch.zeros((), device=z.device)

        # Use entropy loss on the codebook
        if self.entropy_loss_weight != 0.0 and self.training:
            per_sample_entropy, avg_entropy = entropy_loss_fn(-1*d, self.entropy_loss_temperature, self.entropy_gamma)
            entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)

        loss = commitment_loss + codebook_loss + entropy_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            min_encoding_indices=min_encoding_indices.view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3])
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """ Returns the codebook entry for the given indices.
        
        Args:
            indices -> torch.Tensor: The indices of the codebook entries.

        Returns:
            z_quantized -> torch.Tensor: The codebook entries.
        """
        # get quantized latent vectors
        z_quantized = self.embedding(indices.int())
        if self.use_l2_normalisation:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized


def choose_vector_quantizer_class(config):
    if config.quantizer_type == "lookup":
        return SimpleVectorizer(
            config.codebook_size,
            config.token_size,
            config.commitment_cost,
            config.entropy_loss_weight,
            config.entropy_loss_temperature,
            config.entropy_gamma,
            config.get("use_l2_normalisation", False),
        )
    elif config.quantizer_type == "lookup-free":
        return LookupFreeQuantizer(
            config.token_size,
            config.commitment_cost,
            config.entropy_loss_weight,
            config.entropy_loss_temperature,
            config.entropy_gamma,
        )
    else:
        raise ValueError("Unknown vector quantizer class")


class ConvVQModel(BaseModel):
    def __init__(
        self,
        config,
        legacy: bool = False,
        finetune_decoder: bool = False
    ):
        """ Initializes the convolutional VQ-VAE model.

        Args:
            config: The configuration for the model.
            legacy -> bool: Whether to use the legacy decoder.
            finetune_decoder -> bool: Whether to finetune the decoder.
        """
        super().__init__()
        self.config = config
        self.encoder = ConvEncoder(self.config)
        if legacy:
            # To support older weights
            self.decoder = ConvDecoderLegacy(self.config)
        else:
            self.decoder = ConvDecoder(self.config)

        self.finetune_decoder = finetune_decoder
        if self.finetune_decoder:
            self.encoder.eval()
            self.encoder.requires_grad_(False)
        self.quantize = choose_vector_quantizer_class(self.config)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Encodes the input tensor, i.e. runs the encoder.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        z = self.encoder(x)
        z_quantized, result_dict = self.quantize(z)
        return z_quantized, result_dict

    def decode(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """ Decodes the quantized latent representation, i.e. runs the decoder.

        Args:
            z_quantized -> torch.Tensor: The quantized latent representation.

        Returns:
            decoded -> torch.Tensor: The decoded image.
        """
        decoded = self.decoder(z_quantized)
        return decoded

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """ Decodes from tokens, i.e. runs the decoder after converting tokens to latent representations.

        Args:
            tokens -> torch.Tensor: The tokens.

        Returns:
            decoded -> torch.Tensor: The decoded image.
        """
        z_quantized = self.quantize.get_codebook_entry(tokens)
        ss = int(math.sqrt(float(z_quantized.size(1))))
        z_quantized = z_quantized.reshape(z_quantized.size(0), ss, ss, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized)
        return decoded

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Runs the model on the input tensor.

        Args:
            input -> torch.Tensor: The input image.

        Returns:
            decoded -> torch.Tensor: The decoded image.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        if self.finetune_decoder:
            self.encoder.eval()
            z_quantized, result_dict = self._finetuning_encoder_forward(input)
        else:
            z_quantized, result_dict = self.encode(input)

        decoded = self.decode(z_quantized)
        return decoded, result_dict