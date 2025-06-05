"""Model definition.

In this file, we implement the Stage-II model.
"""

import math
from typing import List, Tuple, Union, Optional
from tqdm import tqdm

import torch
from einops import rearrange, repeat, pack, unpack
import einx


# from modeling.modules import BaseModel
    
def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def calc_entropy(logits):
    prob = logits.softmax(dim = -1)
    return (-prob * log(prob)).sum(dim = -1)

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    t, packed_shape = pack([t], pattern)

    def inverse(t, unpack_pattern = None):
        unpack_pattern = default(unpack_pattern, pattern)
        return unpack(t, packed_shape, unpack_pattern)[0]

    return t, inverse


# ============================ Bert Network ============================

class BertFeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0., use_prenorm: bool = False):
        """ Initialize the Multi-Layer Perceptron (MLP).

            Args:
                dim -> int: Dimension of the input tensor.
                hidden_dim -> int: Dimension of the hidden layer.
                dropout -> float: Dropout rate. Defaults to 0.
                use_prenorm -> bool: Flag setting prenorm or postnorm. Defaults to False.
        """
        super(BertFeedForward, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim, bias=True),
            torch.nn.Dropout(dropout),
        )
        self.norm = torch.nn.LayerNorm(dim, eps=1e-12)
        self.use_prenorm = use_prenorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the MLP module.

            Args:
                x -> torch.Tensor: Input tensor.
            Returns:
                torch.Tensor: Output of MLP layer.
        """
        if self.use_prenorm:
            return self._forward_prenorm(x)
        else:
            return self._forward_postnorm(x)

    def _forward_prenorm(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the MLP module with prenorm.

            Args:
                x -> torch.Tensor: Input tensor.
            Returns:
                torch.Tensor: Output of MLP layer.
        """
        y = self.norm(x)
        out = self.net(y)
        return out + x

    def _forward_postnorm(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the MLP module with postnorm.

            Args:
                x -> torch.Tensor: Input tensor.
            Returns:
                torch.Tensor: Output of MLP layer.
        """
        out = self.net(x)
        return self.norm(out + x)


class BertAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., use_prenorm: bool = False):
        """ Initialize the BertAttention module.

            Args:
                embed_dim -> int: Dimension of the input tensor.
                num_heads -> int: Number of heads in the multi-head attention.
                dropout -> float: Dropout rate. Defaults to 0.
                use_prenorm -> bool: Flag setting prenorm or postnorm. Defaults to False.
        """
        super(BertAttention, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.norm = torch.nn.LayerNorm(embed_dim, eps=1e-12)
        self.use_prenorm = use_prenorm

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the BertAttention module.

            Args:
                x -> torch.Tensor: Input tensor.
                return_attn -> bool: Flag setting whether to compute attention weights or not.
                    Setting this to false can enable faster MHSA in pytorch 2.x. Defaults to False.
            Returns:
                (attention_value, attention_weight) -> Tuple[torch.Tensor, torch.Tensor]: 
                    First element is the output of this attention layer, while the second element contain 
                    the attention weights if computed.
        """
        if self.use_prenorm:
            return self._forward_prenorm(x, return_attn)
        else:
            return self._forward_postnorm(x, return_attn)

    def _forward_prenorm(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the BertAttention module with Prenorm.

            Args:
                x -> torch.Tensor: Input tensor.
                return_attn -> bool: Flag setting whether to compute attention weights or not.
                    Setting this to false can enable faster MHSA in pytorch 2.x. Defaults to False.
            Returns:
                (attention_value, attention_weight) -> Tuple[torch.Tensor, torch.Tensor]: 
                    First element is the output of this attention layer, while the second element contain 
                    the attention weights if computed.
        """
        y = self.norm(x)
        attention_value, attention_weight = self.mha(y, y, y, need_weights=return_attn)
        attention_value = self.dropout(attention_value)
        out = attention_value + x

        return out, attention_weight

    def _forward_postnorm(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the BertAttention module with Postnorm.

            Args:
                x -> torch.Tensor: Input tensor.
                return_attn -> bool: Flag setting whether to compute attention weights or not.
                    Setting this to false can enable faster MHSA in pytorch 2.x. Defaults to False.
            Returns:
                (attention_value, attention_weight) -> Tuple[torch.Tensor, torch.Tensor]: 
                    First element is the output of this attention layer, while the second element contain 
                    the attention weights if computed.
        """
        attention_value, attention_weight = self.mha(x, x, x, need_weights=return_attn)
        attention_value = self.dropout(attention_value)
        out = self.norm(attention_value + x)

        return out, attention_weight


class TransformerEncoder(torch.nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float = 0., use_prenorm: bool = False):
        """ Initialize the Transformer module.
        
            Args:
                dim -> int: Dimension of the input tensor.
                depth -> int: Number of attention layers.
                heads -> int: Number of attention heads.
                mlp_dim -> int: Dimension of the MLP.
                dropout -> float: Dropout rate. Defaults to 0.
                use_prenorm -> bool: Flag setting prenorm or postnorm. Defaults to False.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                BertAttention(dim, heads, dropout=dropout, use_prenorm=use_prenorm),
                BertFeedForward(dim, mlp_dim, dropout=dropout, use_prenorm=use_prenorm)
            ]))

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Forward pass through the Attention module.

            Args:
                x -> torch.Tensor: Input tensor.
                return_attn -> bool: Flag setting whether to compute attention weights or not.
                    Setting this to false can enable faster MHSA in pytorch 2.x. Defaults to False.
            Returns:
                (transformer_output, [attention_weights]) -> Tuple[torch.Tensor, List[torch.Tensor]]: 
                    First element is the output of this transformer, while the second element contain 
                    the attention weights if computed.
        """
        l_attn = []
        for attn, ffn in self.layers:
            x, attention_weight = attn(x, return_attn=return_attn)
            x = ffn(x)
            l_attn.append(attention_weight)
        return x, l_attn


class LFQBert(torch.nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int]=256,
        conditional_img_size=512,
        conditional_chans=6,
        instruction_dim=768,
        instruction_len=77,
        hidden_dim=768,
        codebook_size=1024,
        codebook_splits=1,
        depth=24,
        heads=8,
        mlp_dim=3072,
        dropout=0.1,
        # nclass=1000,
        input_stride: int = 16,
        use_prenorm: bool = False,
        train_frac_bits_flipped: float = 0.05
    ):
        """ Initialize the Transformer model.

            Args:
                img_size -> int: The image size. This model expects inputs of size img_size x img_size. Defaults to 256.
                hidden_dim -> int: The hidden dimension. Defaults to 768.
                codebook_size -> int: The codebook size. Defaults to 1024.
                codebook_splits -> int: The number of codebook splits. Defaults to 1.
                depth -> int: The depth of the transformer. Defaults to 24.
                heads -> int: The number of heads in the multi-head attention. Defaults to 8.
                mlp_dim -> int: The MLP dimension. Defaults to 3072.
                dropout -> float: The dropout rate. Defaults to 0.1.
                nclass -> int: The number of classes. Defaults to 1000, which is correct for ImageNet.
                input_stride -> int: The input stride. Defaults to 16.
                use_prenorm -> bool: A Flag setting prenorm or postnorm. Defaults to False.
        """
        super().__init__()
        # self.nclass = nclass
        # self.drop_label = nclass
        if isinstance(img_size, (tuple, list)):
            self.seq_len = int(
                (img_size[0] // input_stride) *
                (img_size[1] // input_stride)
            )
        else:
            self.seq_len = (img_size // input_stride) ** 2
        self.conditional_seq_len = (conditional_img_size // input_stride) ** 2 if conditional_img_size is not None else 0
        self.instruction_len = instruction_len if instruction_len is not None else 0
        self.splits = codebook_splits
        self.bits = int(math.log2(codebook_size))
        effective_bits = self.bits // self.splits
        self.effective_codebook_size = int(2 ** effective_bits)
        self.mask_token = 0 # self.effective_codebook_size
        self.train_frac_bits_flipped = train_frac_bits_flipped
        bits_to_indices = torch.pow(2.0, torch.arange(0, effective_bits))
        self.register_buffer('bits_to_indices', bits_to_indices.int())
        
        # =================================== layers ===================================
        
        # instruction embedding
        self.intruction_proj = torch.nn.Linear(instruction_dim, hidden_dim)

        # Required by the task of class-conditional generation. 
        # self.class_proj = torch.nn.Embedding(nclass+1, hidden_dim) # +1 for class drop

        self.token_proj = torch.nn.Linear(self.bits, hidden_dim)
        self.conditional_token_proj = torch.nn.Linear(conditional_chans, hidden_dim)
        # self.uni_proj = torch.nn.Linear(hidden_dim, hidden_dim)

        # positional embedding
        self.pos_emb = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, self.seq_len, hidden_dim)), 0., 0.02)
        self.conditional_pos_emb = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, self.conditional_seq_len, hidden_dim)), 0., 0.02) if self.conditional_seq_len > 0 else None
        self.instruction_pos_emb = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, self.instruction_len, hidden_dim)), 0., 0.02) if self.instruction_len > 0 else None
        
        # First layer before the Transformer block
        self.first_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim, eps=1e-12),
            torch.nn.Dropout(p=dropout)
        )

        self.use_prenorm = use_prenorm
        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, use_prenorm=use_prenorm)
        if self.use_prenorm:
            self.norm_after_transformer = torch.nn.LayerNorm(hidden_dim, eps=1e-12)
        
        # Last layer after the Transformer block
        self.last_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim, eps=1e-12),
        )

        # k * 2
        self.prediction_layer = torch.nn.Linear(hidden_dim, self.bits * 2)

        self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module):
        """ Initialize the weights.

            Args:
                module -> torch.nn.Module: The module to initialize.
        """
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data = torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_group_splits(self) -> int:
        return self.splits

    def preprocess_tokens(self, img_tokens: torch.Tensor) -> torch.Tensor:
        """ Preprocess the tokens by converting from indices to {-1,1} bits and setting masked area to 0.

            Args:
                img_tokens -> torch.Tensor: The image tokens.
            
            Returns:
                torch.Tensor: The preprocessed image tokens.
        """
        # img_tokens: [bs, l, m]
        mask = img_tokens == self.mask_token
        token_as_bits = ((img_tokens[..., None].int() & self.bits_to_indices) != 0).float()
        # convert to {-1, 1}
        token_as_bits = token_as_bits * 2.0 - 1.0
        # mask tokens to be zero
        token_as_bits[mask, :] = torch.full_like(token_as_bits[mask, :], 0.0)
        # m is 2
        token_as_bits = rearrange(token_as_bits, "b n m c -> b n (m c)")
        return token_as_bits
    
    def interpolate_pos_emb(self, pos_embd: torch.Tensor, seq_len: int) -> torch.Tensor:
        """ Interpolate the positional embedding.
        """
        # [bs, l, c]
        pt_seq_len = pos_embd.shape[1]
        if pt_seq_len != seq_len:
            pos_embd = torch.nn.functional.interpolate(
                pos_embd.permute(0, 2, 1),
                size=(seq_len,),
                mode='linear',
                align_corners=False
            )
            pos_embd = pos_embd.permute(0, 2, 1)
        return pos_embd
    
    def forward(
        self,
        img_tokens: torch.Tensor,
        img_cond_tokens: torch.Tensor | tuple[torch.Tensor] | None=None,
        degradation_instruction: torch.Tensor | None=None,
        return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward pass.

            Args:
                img_tokens -> torch.Tensor: The image tokens.
                class_labels -> torch.Tensor: The class labels.
                drop_label_mask -> Optional[torch.Tensor]: The mask for the drop label. Defaults to None.
                return_attn -> bool: Flag setting whether to compute attention weights or not. Defaults to False.
            
            Returns:
                Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
                    If return_attn is False, returns the output Tensor of shape (b, n, m, c), where n is the sequence length,
                    m is the number of splits, and c is the effective codebook size.
                    If return_attn is True, returns a tuple of (torch.Tensor, List[torch.Tensor]), where the first item is 
                    the output tensor described above and the second item is a list of attention weights.
        """
        b, outp_len = img_tokens.shape[:2]
        
        # handle img_cond_tokens
        if img_cond_tokens is not None:
            # img_cond_tokens = self.preprocess_tokens(img_cond_tokens)
            if isinstance(img_cond_tokens, torch.Tensor):
                img_cond_tokens = self.conditional_token_proj(img_cond_tokens)
            # share one positional embedding
            elif isinstance(img_cond_tokens, tuple):
                img_cond_tokens = self.conditional_token_proj(torch.cat(img_cond_tokens, dim=-1))
            img_cond_tokens = img_cond_tokens + self.interpolate_pos_emb(self.conditional_pos_emb, img_cond_tokens.shape[1])
                        
        if degradation_instruction is not None:
            degradation_instruction = self.intruction_proj(degradation_instruction)
            degradation_instruction = degradation_instruction + self.interpolate_pos_emb(self.instruction_pos_emb, degradation_instruction.shape[1])

        # input is codes not indices, it's different from the GPT2 AR models
        # or says AR models' inputs are indices, and convert to another code by an embedding layer
        # k -> hidden_dim, shape as (bs, n x m, c) -> (bs, n x m, hidden_dim)
        token_feature = self.token_proj(img_tokens)

        # add class token to the first token
        # token_feature = torch.cat([token_feature, cls_feature], dim=1)

        # Position embedding
        pos_embeddings = self.interpolate_pos_emb(self.pos_emb, img_tokens.shape[1])
        x = token_feature + pos_embeddings
        
        # cat x, img_cond_tokens, and degradation_instruction
        if img_cond_tokens is not None:
            x = torch.cat([x, img_cond_tokens], dim=1)
        if degradation_instruction is not None:
            x = torch.cat([x, degradation_instruction], dim=1)
        # x = self.uni_proj(x)
        
        # TODO: add rope positional embedding in attention

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x, return_attn=return_attn)
        if self.use_prenorm:
            x = self.norm_after_transformer(x)
        x = self.last_layer(x)

        # to [bs, l * k, 2]
        logits = self.prediction_layer(x)[:, :outp_len]
        logits = rearrange(logits, "b l (k bits) -> b (l k) bits", bits=2)        

        if return_attn:  # return list of attention
            return logits, attn

        return logits
    
    def forward_loss(
        self,
        img_tokens: torch.Tensor,
        img_cond_tokens: torch.Tensor | tuple[torch.Tensor] | None=None,
        degradation_instruction: torch.Tensor | None=None,
        return_attn: bool = False,
    ):
        # flip some bits
        if self.train_frac_bits_flipped > 0.:
            num_bits_to_flip = img_tokens.shape[1] * self.train_frac_bits_flipped
            # for every bits, flip_mask: [bs, l * k]
            img_tokens, inverse = pack_one(img_tokens, 'b *')
            flip_mask = torch.rand_like(img_tokens).argsort(dim = -1) < num_bits_to_flip
            img_tokens = torch.where(flip_mask, img_tokens * -1, img_tokens)
            img_tokens = inverse(img_tokens)
        else:
            flip_mask = torch.zeros_like(img_tokens, dtype=torch.bool)
        
        # split m
        orig_codes = rearrange(img_tokens.clone(), 'b l k -> b (l k)')
        img_tokens = rearrange(img_tokens, 'b l (m k_div_m) -> b (l m) k_div_m', m=self.splits)
        
        # img_tokens: [bs, l * m, k // m]
        bs, seq_len = img_tokens.shape[:2]
        
        # mask encoded image codes
        times = torch.rand(bs, device=img_tokens.device)
        times = torch.cos(times * math.pi * 0.5)
        n_masked = (seq_len * times).ceil().clamp(min=1)
        masks = torch.rand(bs, seq_len, device=img_tokens.device).argsort(dim=-1) < n_masked.unsqueeze(-1)
        
        # mask image tokens
        img_tokens = einx.where('bs l_x_m, , bs l_x_m k_div_m -> bs l_x_m k_div_m', masks, 0., img_tokens)
        img_tokens = rearrange(img_tokens, 'b (l m) k_div_m -> b l (m k_div_m)', m=self.splits)
    
        # [bs, l * k, 2]
        logits = self.forward(img_tokens, 
                              img_cond_tokens, 
                              degradation_instruction,
                              return_attn=return_attn)
        
        # repeat binary labels
        masks = repeat(masks, 'b l_x_m -> b (l_x_m k_div_m)', k_div_m=self.bits // self.splits).bool()
        masks = masks | flip_mask
        labels = (orig_codes[masks] > 0).long()
        
        # loss and accuracy
        loss = torch.nn.functional.cross_entropy(logits[masks], labels, ignore_index=-1)
        acc = (logits[masks].argmax(dim=-1) == labels).float().mean()
        
        return dict(logits=logits, loss=loss, correct_tokens=acc)
    
    def forward_loss_random(
        self,
        img_tokens: torch.Tensor,
        img_cond_tokens: torch.Tensor | tuple[torch.Tensor] | None=None,
        degradation_instruction: torch.Tensor | None=None,
        return_attn: bool = False,
    ):
        # split m
        img_tokens = rearrange(img_tokens, 'b l k -> b (l k)')
        orig_codes = img_tokens.clone()
        
        # img_tokens: [bs, l * k]
        bs, seq_len = img_tokens.shape[:2]
        
        # mask encoded image codes
        times = torch.rand(bs, device=img_tokens.device)
        times = torch.cos(times * math.pi * 0.5)
        n_masked = (seq_len * times).ceil().clamp(min=1)
        masks = torch.rand(bs, seq_len, device=img_tokens.device).argsort(dim=-1) < n_masked.unsqueeze(-1)
        
        # mask image tokens
        img_tokens = einx.where('bs l_x_k, , bs l_x_k -> bs l_x_k', masks, 0., img_tokens)
        img_tokens = rearrange(img_tokens, 'b (l k) -> b l k', k=self.bits)
    
        # [bs, l * k, 2]
        logits = self.forward(img_tokens, 
                              img_cond_tokens, 
                              degradation_instruction,
                              return_attn=return_attn)
        
        # get labels
        masks = masks.bool()
        labels = (orig_codes[masks] > 0).long()
        
        # loss and accuracy
        loss = torch.nn.functional.cross_entropy(logits[masks], labels, ignore_index=-1)
        acc = (logits[masks].argmax(dim=-1) == labels).float().mean()
        
        return dict(logits=logits, loss=loss, correct_tokens=acc)
    
    
    @torch.no_grad()
    def sample(self,
               vq_model: torch.nn.Module,
               conditions: tuple[torch.Tensor] | None = None,
               num_samples: int = 10,
               num_steps: int = 12,
               img_size: tuple[int, int] = (32, 32),
               codebook_size: int = 2 ** 18,
               codebook_splits: int = 2,
               gumbel_temperature: float = 0.5,
        ):
        device = 'cuda'  # accelerator assumes the default device to be 'cuda'
        vq_model.eval()
        
        spatial_size = int(img_size[0] * img_size[1])
        num_splits = int(codebook_splits)
        codebook_dim = int(math.log2(codebook_size))
        # [bs, l * k]
        masked_codes = torch.zeros((num_samples, spatial_size * codebook_dim), device=device)

        # progress
        times = torch.linspace(0., 1., num_steps, device = device)
        noise_levels = torch.cos(times * torch.pi * 0.5)
        num_bits_to_mask = (noise_levels * masked_codes.shape[1]).long().ceil().clamp(min = 1)
        traj_codes = []
        
        for (idx, bits_to_mask), progress in tqdm(zip(enumerate(num_bits_to_mask), reversed(range(num_steps))), 
                                                total=num_steps,
                                                leave=False, desc=f'sampling for {num_steps} steps ...'):
            is_first = idx == 0
            
            if not is_first:
                entropy = calc_entropy(logits)
                remask_indices = entropy.topk(bits_to_mask.item(), dim=-1).indices
                masked_codes.scatter_(1, remask_indices, 0.) # recall they use 0. for masking

            # to model
            model_inp = rearrange(masked_codes, 'b (l k) -> b l k', l=spatial_size).float()
            # [bs, l * k, 2]
            if conditions is not None:
                logits = self.forward(model_inp, *conditions)
            else:
                logits = self.forward(model_inp)
            
            # sample
            gumbel_temperature = gumbel_temperature * progress / num_steps  # annealing
            masked_codes = gumbel_sample(logits, temperature=gumbel_temperature)
            masked_codes = masked_codes * 2 - 1
            
            # to traj_codes
            traj_codes.append(
                rearrange(masked_codes, 'b (l k) -> b l k', l=spatial_size)
                .detach().cpu()
            )
            
        # decode
        masked_codes = rearrange(masked_codes, 'b (l k) -> b l k', l=spatial_size)
        pred_imgs = vq_model.decode_tokens(masked_codes)
        
        return pred_imgs, traj_codes
        
    
if __name__ == "__main__":
    model = LFQBert(img_size=512, input_stride=16, codebook_size=2**18, codebook_splits=2).cuda()
    # lfq tokens
    bs = 1
    factorized_n = 2
    img_tkz_size = int(512 // 16)
    seq_len = img_tkz_size ** 2
    lfq_codes = torch.randint(0, 2, (bs, seq_len, 18)).cuda()
    lfq_factorized_codes = rearrange(lfq_codes, "b l (m c) -> b (l m) c", m=factorized_n)
    factorized_seq_len = lfq_factorized_codes.shape[1]
    
    #* mask some codes
    # from scipy.stats import truncnorm
    # import numpy as np
    
    # mask_ratio_min = 0.7
    # mask_ratio_max = 1.0
    # mask_dist = truncnorm(a=(mask_ratio_min - 1.0) / 0.25, b=0, loc=1, scale=0.25)
    # mask_ratio = mask_dist.rvs(size=1)[0]
    # n_masked_tokens = int(np.ceil(mask_ratio * factorized_seq_len))
    # mask = torch.zeros(bs, factorized_seq_len).to(lfq_factorized_codes)
    # # random order
    # orders = torch.stack([torch.randperm(factorized_seq_len).cuda().long() for _ in range(bs)], dim=0)
    # # mask n_masked_tokens tokens
    # mask = torch.scatter(mask, dim=-1, index=orders[:, :n_masked_tokens],
    #                      src=torch.ones(bs, n_masked_tokens).to(lfq_factorized_codes))
    # # mask lfq codes
    # # to [-1, 1]
    # lfq_factorized_codes = lfq_factorized_codes * 2.0 - 1.0
    # lfq_factorized_codes = lfq_factorized_codes * (1 - mask.unsqueeze(-1))
    # print(lfq_factorized_codes.shape)
    # lfq_factorized_codes = rearrange(lfq_factorized_codes, "b (l m) c -> b l (m c)", m=factorized_n)
    
    
    #* mask function in trainer ==============================================
    from torch import Tensor
    import math
    import numpy as np
    
    #* mask the input codes after lfq =========================================
    def mask_inp_codes(inp_codes: Tensor, split_m: int):
        bs, seq_len, codebook_k = inp_codes.size()
        device = inp_codes.device
        assert codebook_k % split_m == 0, f'codebook_k: {codebook_k} should be divisible by split_m: {split_m}'
        seq_len = int(np.ceil(seq_len * split_m))
        
        # mask some codes
        timesteps = torch.zeros((bs,), device=device).float().uniform_(0, 1.0)
        mask_ratio = torch.acos(timesteps) / (math.pi * 0.5) # arccos schedule
        mask_ratio = torch.clamp(mask_ratio, min=1e-6, max=1.)
        num_token_masked = (seq_len * mask_ratio).round().clamp(min=1)
        batch_randperm = torch.rand(bs, seq_len, device=device).argsort(dim=-1)
        masks = (batch_randperm < rearrange(num_token_masked, 'b -> b 1')).type(torch.float32)
        
        # factorize and mask codes
        factorized_codes = rearrange(inp_codes, 'b l (m c) -> b (l m) c', m=split_m)
        factorized_codes = factorized_codes * (1 - masks.unsqueeze(-1))
        factorized_codes = rearrange(factorized_codes, 'b (l m) c -> b l (m c)', m=split_m)
        
        return factorized_codes, masks
    
    lfq_codes = lfq_codes * 2 - 1
    lfq_factorized_codes, masks = mask_inp_codes(lfq_codes, factorized_n)
    
    #* to model ==============================================
    # logits: [bs, (l * k), 2]
    logits = model(lfq_factorized_codes)
    
    
    #* set loss ================================================
    import einops
    import torch.nn.functional as F
    # from utils.loss_utils import MLMLoss
    
    # [bs, l * m] -> [bs, l * m * (k // m)] = [bs, l * k]
    mask_rep = einops.repeat(masks, 'b lxm -> b (lxm k_div_m)', k_div_m=18 // factorized_n).bool()
    orig_codes = rearrange(lfq_codes, 'b l k -> b (l k)')
    labels = (orig_codes[mask_rep] > 0).long() # {0, 1}
    
    loss = F.cross_entropy(logits[mask_rep], labels, ignore_index=-1)
    print('maskbit loss: ', loss)
    
    #* sampling ===============================================
    from typing import Text
    
    # def get_masking_ratio(progress: float, mode: Text = "arccos") -> torch.Tensor:
    #     """ Get masking ratio. """
    #     r = torch.tensor(progress)
    #     if mode == "root":
    #         val_to_mask = 1 - (r ** 0.5)
    #     elif mode == "square":
    #         val_to_mask = 1 - (r ** 2)
    #     elif mode == "cosine":
    #         val_to_mask = torch.cos(r * math.pi * 0.5)
    #     elif mode == "arccos":
    #         val_to_mask = torch.acos(r) / (math.pi * 0.5)
    #     elif mode == "linear":
    #         val_to_mask = 1 - r
    #     else:
    #         raise ValueError("Invalid mode. Choose between 'linear','square', 'cosine', 'arccos', 'root'.")
        
    #     val_to_mask = torch.clamp(val_to_mask, 1e-6, 1.0)
    #     return val_to_mask

    # def combine_factorized_tokens(tokens: Tensor, codebook_size: int, splits: int):
    #     # [bs, n]
    #     combined_tokens = torch.zeros((tokens.shape[0], tokens.shape[1]), device=tokens.device)
    #     bit_shift = int(math.log2(codebook_size)) // splits  # 2 ** k = codebook_size -> bit_shift = k / splits, by default, k = 8
    #     for i in range(splits):
    #         # e.g., k=6, m=2
    #         # 0,0,1,1,0,1 = 13
    #         # 0,0,1 << 1 * 3 = 8
    #         # 1,0,1 << 0 * 3 = 5
            
    #         combined_tokens += (tokens[..., i] << (i * bit_shift))
    #     return combined_tokens

    # peusodu VQ VAE
    
    class VQVAE(torch.nn.Module):
        def __init__(self, codebook_size: int, splits: int):
            super().__init__()
            self.codebook_size = codebook_size
            self.splits = splits
                
        def decode_tokens(self, tokens: Tensor):
            return tokens
        
    vqvae = VQVAE(codebook_size=2**18, splits=2).cuda()
            
    # @torch.no_grad()
    # def sample(
    #     model,
    #     vqgan_model,
    #     num_samples: int = 10,
    #     # labels: Optional[torch.Tensor] = None,
    #     conditions: Optional[Union[Tensor, list[Tensor]]] = None,
    #     softmax_temperature: float = 1.0,
    #     randomize_temperature: float = 4.5,
    #     mask_schedule_strategy: Text = "linear",
    #     num_steps: int = 12,
    #     mask_token: int = 0,
    #     patch_size: int = 16,
    #     # guidance_scale: float = 0.,
    #     # guidance_annealing: Text = "none",
    #     # scale_pow: float = 4.0,
    #     codebook_size: int = 2 ** 18,
    #     codebook_splits: int = 2,
    # ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    #     device = next(model.parameters()).device
    #     model.eval()
    #     vqgan_model.eval()

    #     # drop_labels = torch.ones(num_samples, dtype=bool, device=device)
    #     spatial_size = int(patch_size ** 2)
    #     num_splits = int(codebook_splits)
    #     # [bs, l, m]
    #     masked_tokens = torch.full((num_samples, spatial_size, num_splits), mask_token, 
    #                                device=device)
    #     # [bs, l, k]
    #     codebook_dim = int(math.log2(codebook_size))
    #     masked_codes = torch.full((num_samples, spatial_size, codebook_dim), 
    #                               mask_token, dtype=torch.float32, device=device)
        
    #     # l * m
    #     num_maskable = spatial_size * num_splits
    #     mask = torch.ones(num_samples, spatial_size, num_splits, dtype=torch.bool, device=device)
    #     num_sampled = torch.zeros_like(masked_tokens, dtype=torch.int)
    #     l_full_tokens = []
    #     gumbel = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    #     # handle None conditions
    #     if conditions is None:
    #         conditions = (None, None)
            
    #     from tqdm import trange

    #     for i in trange(num_steps, leave=False, desc=f'sampling for {num_steps} steps ...'):
    #         progress = (i + 1) / num_steps
            
    #         # we do not use guidance here
            
    #         # if guidance_scale != 0.0:
    #         #     logits = model(
    #         #         torch.cat([masked_tokens.clone(), masked_tokens.clone()], dim=0),
    #         #         torch.cat([labels, labels], dim=0),
    #         #         torch.cat([~drop_labels, drop_labels], dim=0)
    #         #     )
    #         #     # Classifier-free guidance
    #         #     logits_with_class, logits_without_class = torch.chunk(logits, 2, dim=0)
    #         #     if guidance_annealing == "none":
    #         #         scale_step = 1.0
    #         #     elif guidance_annealing == "linear":
    #         #         scale_step = i / num_steps
    #         #     elif guidance_annealing == "cosine":
    #         #         scale_pow = torch.ones((1), device=device) * scale_pow
    #         #         scale_step = (1 - torch.cos(((i / num_steps) ** scale_pow) * torch.pi)) * 1/2 # power-cos scaling
    #         #     scale = guidance_scale * scale_step
    #         #     logits = logits_with_class + scale * (logits_with_class - logits_without_class)
    #         # else:
    #         #     logits = model(masked_tokens.clone(), labels, ~drop_labels)
            
    #         # logits: [bs, l, m, 2 ** (k // m)]
    #         logits = model(masked_codes, *conditions)
            
    #         # softmax temperature
    #         # TODO: may annealing here
    #         probabilities = torch.softmax(logits / softmax_temperature, dim=-1)
            
    #         # categorical distribution
    #         distribution = torch.distributions.Categorical(probabilities)
    #         predicted_tokens = distribution.sample()  # [bs, l, m]
            
    #         # mask: [bs, l, m] -> [bs,]
    #         num_masked = torch.sum(mask, dim=(1, 2))[0]

    #         # replace masked tokens with predicted tokens
    #         # predicted_tokens: [bs, l, m]
    #         predicted_tokens = torch.where(mask, predicted_tokens, masked_tokens)
            
    #         # confidence
    #         # gather([bs, l, m, 2 ** (k // m)], -1, [bs, l, m, 1]) -> [bs, l, m]
    #         confidence = torch.gather(probabilities, -1, predicted_tokens.unsqueeze(-1)).squeeze(-1)
            
    #         # Ignore existing tokens by overwriting the confidence.
    #         confidence = torch.where(mask, confidence, torch.inf)

    #         # gumbel noise
    #         noise = gumbel.sample(predicted_tokens.size()) * randomize_temperature * (1 - progress)
    #         confidence = torch.log(confidence) + noise.to(device)

    #         # masking ratio
    #         mask_ratio = get_masking_ratio(progress, mode=mask_schedule_strategy).to(device)
            
    #         # min = 1, max = num_masked - 1
    #         mask_len = torch.floor(mask_ratio * num_maskable)
    #         num_tokens_to_mask = torch.clamp(mask_len, torch.ones_like(num_masked), num_masked-1).long()
    #         sorted_confidence = torch.sort(confidence.view(num_samples, -1), dim=-1).values
    #         threshold = sorted_confidence[:, num_tokens_to_mask - 1]

    #         should_mask = (confidence <= threshold.unsqueeze(-1).unsqueeze(-1))
    #         masked_tokens = torch.where(should_mask, mask_token, predicted_tokens)
    #         mask = (masked_tokens == mask_token)
    #         num_sampled += torch.where(should_mask, 0, 1)
    #         l_full_tokens.append(predicted_tokens)
            
    #         # update masked_codes
    #         masked_codes = model.preprocess_tokens(masked_tokens)

    #     predicted_tokens = combine_factorized_tokens(predicted_tokens, codebook_size, codebook_splits)
    #     generated_image = vqgan_model.decode_tokens(predicted_tokens)
    #     return generated_image, l_full_tokens
        
        
    #* sampling ver2 ================================================
    from tqdm import tqdm
    
    def sample_v2(
        model,
        vq_model,
        conditions: tuple[torch.Tensor] | None = None,
        num_samples: int = 10,
        num_steps: int = 12,
        mask_token: int = 0,
        img_size: tuple[int, int] = (32, 32),
        codebook_size: int = 2 ** 18,
        codebook_splits: int = 2,
        gumbel_temperature: float = 1.0,
    ):
        device = next(model.parameters()).device
        model.eval()
        vq_model.eval()
        
        spatial_size = int(img_size[0] * img_size[1])
        num_splits = int(codebook_splits)
        codebook_dim = int(math.log2(codebook_size))
        # [bs, l * m * k]
        masked_codes = torch.zeros((num_samples, spatial_size * codebook_dim), device=device)
        seq_len = masked_codes.shape[1]
        
        # progress
        times = torch.linspace(0., 1., num_steps, device = device)
        noise_levels = torch.cos(times * torch.pi * 0.5)
        num_bits_to_mask = (noise_levels * seq_len).long().ceil().clamp(min = 1)
        
        for idx, bits_to_mask in tqdm(enumerate(num_bits_to_mask)):
            is_first = idx == 0
            
            if not is_first:
                entropy = calc_entropy(logits)
                remask_indices = entropy.topk(bits_to_mask.item(), dim=-1).indices
                masked_codes.scatter_(1, remask_indices, 0.) # recall they use 0. for masking

            # to model
            model_inp = rearrange(masked_codes, 'b (l k) -> b l k', l=spatial_size).float()
            # [bs, l * k, 2]
            logits = model(model_inp)
            
            # sample
            masked_codes = gumbel_sample(logits, temperature=gumbel_temperature)
            masked_codes = masked_codes * 2 - 1
            
        # decode
        masked_codes = rearrange(masked_codes, 'b (l k) -> b l k', l=spatial_size)
        pred_imgs = vq_model.decode_tokens(masked_codes)
        
        return pred_imgs
    
    # sample it
    # gen_img, _ = sample(
    #     model,
    #     vqvae,
    #     num_samples=1,
    #     num_steps=100,
    # )
    
    print('sampling codes ...')
    gen_img = sample_v2(
        model,
        vqvae,
        num_samples=1,
        num_steps=100,
    )
    
    print(gen_img.shape)
