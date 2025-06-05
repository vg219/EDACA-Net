import os
import os.path as osp
import math
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed
from einops import rearrange
import accelerate
from accelerate import PartialState
from accelerate.utils import DataLoaderConfiguration, ProjectConfiguration, GradientAccumulationPlugin
from typing import Dict, List, Optional, Text, Tuple, Union, Generator
from dataclasses import dataclass
from torchvision.utils import make_grid
import warnings

from model.lfq_vae.autoencoder import Decoder, Encoder
from model.lfq_vae.quantizer import LFQ_v0 as LFQ
from model.maskbit.maskbit_model import LFQBert as MaskGit
from utils import (
    NameSpace,
    EMA,
    EasyProgress,
    BestMetricSaveChecker,
    TensorboardLogger,
    NoneLogger,
    MetricsByTask,
    AnalysisFusionAcc,
    get_optimizer,
    get_scheduler,
    get_train_dataset,
    get_loss,
)

@dataclass
class BatchedInput:
    image: Tensor  # fused image
    instruction: Tensor  # degradation encoded instruction
    
    image_codes: Tensor = None  # encoded fused image codes
    source1_codes: Tensor = None  # encoded visible/over/far codes
    source2_codes: Tensor = None # encoded hidden/under/near codes
    
    source1: Tensor = None  # degraded visible/over/far
    source2: Tensor = None  # degraded hidden/under/near
    
    image_ids: Tensor = None  # encoded fused image ids
    source1_ids: Tensor = None  # encoded visible/over/far ids
    source2_ids: Tensor = None  # encoded hidden/under/near ids
    
    source1_clean: Tensor = None  # clean visible/over/far
    source2_clean: Tensor = None  # clean hidden/under/near
    
    def __post_init__(self):
        # accelerate can not handle webdataset.DataPipeline
        for k, v in self.__dict__.items():
            if v is not None and isinstance(v, Tensor):
                self.__dict__[k] = v.float().cuda()

    
#========================================================================================================================
# 0. dataset prepare
# 0.1 fused image / degradation (instruction): lighted the image `+2ev`


##* options
# 1. maskgit -  discrerte diffsion | transformer
# 2. next-token prediction transformer (open-magvit 2)
# 3. discrete flow matching | model | DiT

##* 
# 4. vqgan (lfq) img -> codebook -> quantized image ([-1, 1])   | shape: (3, 256, 256) -> (18, 16, 16) | error of image
#           ------------>  lfq + diffusion (flow matching - continuous)  diffusion is lighting (mlp) shaped as (3, 256, 256)

# ----------- MAR, kaiming he --------------
# ----------- VAR + diffusion --------------, var vq_quantizer + diffusion
# 4.1 img -> vq -> maskgit / diffusion / transformer (18 x 16 x 16) -> decoder (3, 256, 256) -> diffusion | flow matching (continuous) (3 x 256 x 256)
#                 | computation heavy |                                                        | computation light

# 5. 
# enc: encoder
# d: degradation
# instruction (text, 'relight') ---> T5/FLIP/SigLIP/charater-level-T5 ------> instruction_embedding (1 x 512 x 256): 512 is the hidden size, 256 is the max length (length of instruction)
# model(enc_s1_d, enc_s2_d, instruction_embedding) -> enc_fused_image (1 x 18 x 16 x 16) (-1, 1)
# -> group -> (1, 2, 9, 16, 16)

# maskgit:
# loss: entropy loss, kl divergence (2 ** 18); group !!

#========================================================================================================================

class StepsCounter:
    def __init__(self):
        self.n_train_steps = 0
        self.n_val_steps = 0
        
    def state_dict(self):
        return {"n_train_steps": self.n_train_steps, "n_val_steps": self.n_val_steps}
    
    def load_state_dict(self, state_dict):
        self.n_train_steps = state_dict["n_train_steps"]
        self.n_val_steps = state_dict["n_val_steps"]
    
    def update(self, mode='train'):
        if mode == 'train':
            self.n_train_steps += 1
        elif mode == 'val':
            self.n_val_steps += 1
        else:
            raise ValueError(f'Invalid mode: {mode}')
        
def pack_to_1d_seq(x: Tensor):
    return rearrange(x, 'b c ... -> b (...) c')

def make_batch_input(batch: Union[Dict, List, Tuple]):
    if isinstance(batch, dict):
        return BatchedInput(
            image_codes=pack_to_1d_seq(batch['encoded_gt']),
            source1_codes=pack_to_1d_seq(batch['encoded_m1_degraded']),
            source2_codes=pack_to_1d_seq(batch['encoded_m2_degraded']),
            instruction=batch['encoded_instruction'],
            image=batch['gt'] if 'gt' in batch else None,
            # degraded images
            source1=batch['m1_degraded'] if 'm1_degraded' in batch else None,
            source2=batch['m2_degraded'] if 'm2_degraded' in batch else None,
            # saved vq ids
            source1_ids=batch['m1_ids'] if 'm1_ids' in batch else None,
            source2_ids=batch['m2_ids'] if 'm2_ids' in batch else None,
            ## clean images
            source1_clean=batch['m1_clean'] if 'm1_clean' in batch else None,
            source2_clean=batch['m2_clean'] if 'm2_clean' in batch else None,
        )
    elif isinstance(batch, (tuple, list)):
        # assume batch is a list of [image_ids (gt), source1_ids (m1_ids), source2_ids (m2_ids), instruction]
        match len(batch):
            case 4:
                return BatchedInput(
                    image_codes=pack_to_1d_seq(batch[0]),
                    source1_codes=pack_to_1d_seq(batch[1]),
                    source2_codes=pack_to_1d_seq(batch[2]),
                    instruction=batch[3],
                )
            case 6:
                return BatchedInput(
                    image_codes=pack_to_1d_seq(batch[0]),
                    source1_codes=pack_to_1d_seq(batch[1]),
                    source2_codes=pack_to_1d_seq(batch[2]),
                    instruction=batch[3],
                    source1=batch[4],
                    source2=batch[5],
                )
            case 7:
                return BatchedInput(
                    image_codes=pack_to_1d_seq(batch[0]),
                    source1_codes=pack_to_1d_seq(batch[1]),
                    source2_codes=pack_to_1d_seq(batch[2]),
                    instruction=batch[3],
                    source1=batch[4],
                    source2=batch[5],
                    image=batch[6], # fused image in pixel space
                )
            case _:
                raise ValueError(f'Invalid batch length: {len(batch)}')
    else:
        raise ValueError(f'Invalid batch type: {type(batch)}')
    
def get_masking_ratio(progress: float, mode: Text = "arccos") -> torch.Tensor:
    """ Get masking ratio. """
    r = torch.tensor(progress)
    if mode == "root":
        val_to_mask = 1 - (r ** 0.5)
    elif mode == "square":
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":
        val_to_mask = torch.acos(r) / (math.pi * 0.5)
    elif mode == "linear":
        val_to_mask = 1 - r
    else:
        raise ValueError("Invalid mode. Choose between 'linear','square', 'cosine', 'arccos', 'root'.")
    
    val_to_mask = torch.clamp(val_to_mask, 1e-6, 1.0)
    return val_to_mask

        
def combine_factorized_tokens(tokens: Tensor, codebook_size: int, splits: int):
    # [bs, n]
    combined_tokens = torch.zeros((tokens.shape[0], tokens.shape[1]), dtype=torch.int, device=tokens.device)
    bit_shift = int(math.log2(codebook_size)) // splits  # 2 ** k = codebook_size -> bit_shift = k / splits, by default, k = 8
    for i in range(splits):
        # e.g., k=6, m=2
        # 0,0,1,1,0,1 = 13
        # 0,0,1 << 1 * 3 = 8
        # 1,0,1 << 0 * 3 = 5
        
        combined_tokens += (tokens[..., i] << ((splits - i - 1) * bit_shift))
    return combined_tokens
        
@torch.no_grad()
def sample(
    model: torch.nn.Module,
    vqgan_model: torch.nn.Module | object,  # module using VQ Taming Model or object has `decode_tokens` method
    num_samples: int = 10,
    # labels: Optional[torch.Tensor] = None,
    conditions: Optional[Union[Tensor, list[Tensor]]] = None,
    softmax_temperature: float = 1.0,
    randomize_temperature: float = 0.1, #4.5,
    mask_schedule_strategy: Text = "arccos",
    num_steps: int = 12,
    mask_token: int = 0,
    sample_size: list[int] = [16, 16],
    # guidance_scale: float = 0.,
    # guidance_annealing: Text = "none",
    # scale_pow: float = 4.0,
    codebook_size: int = 2 ** 18,
    codebook_splits: int = 2,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    device = next(model.parameters()).device

    model.eval()
    vqgan_model.eval()

    # drop_labels = torch.ones(num_samples, dtype=bool, device=device)
    spatial_size = int(math.prod(sample_size))
    num_splits = int(codebook_splits)
    # [bs, l, m]
    masked_tokens = torch.full((num_samples, spatial_size, num_splits), mask_token, 
                                device=device)
    # [bs, l, k]
    codebook_dim = int(math.log2(codebook_size))
    masked_codes = torch.full((num_samples, spatial_size, codebook_dim), 
                                mask_token, dtype=torch.float32, device=device)
    
    # l * m
    num_maskable = spatial_size * num_splits
    mask = torch.ones(num_samples, spatial_size, num_splits, dtype=torch.bool, device=device)
    num_sampled = torch.zeros_like(masked_tokens, dtype=torch.int)
    l_full_tokens = []
    gumbel = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    for i in range(num_steps):
        progress = (i + 1) / num_steps
        
        # we do not use guidance here
        
        # if guidance_scale != 0.0:
        #     logits = model(
        #         torch.cat([masked_tokens.clone(), masked_tokens.clone()], dim=0),
        #         torch.cat([labels, labels], dim=0),
        #         torch.cat([~drop_labels, drop_labels], dim=0)
        #     )
        #     # Classifier-free guidance
        #     logits_with_class, logits_without_class = torch.chunk(logits, 2, dim=0)
        #     if guidance_annealing == "none":
        #         scale_step = 1.0
        #     elif guidance_annealing == "linear":
        #         scale_step = i / num_steps
        #     elif guidance_annealing == "cosine":
        #         scale_pow = torch.ones((1), device=device) * scale_pow
        #         scale_step = (1 - torch.cos(((i / num_steps) ** scale_pow) * torch.pi)) * 1/2 # power-cos scaling
        #     scale = guidance_scale * scale_step
        #     logits = logits_with_class + scale * (logits_with_class - logits_without_class)
        # else:
        #     logits = model(masked_tokens.clone(), labels, ~drop_labels)
        
        # logits: [bs, l, m, 2 ** (k // m)]
        logits = model(masked_codes, *conditions)
        
        # softmax temperature
        # TODO: may annealing here
        probabilities = torch.softmax(logits / softmax_temperature, dim=-1)
        
        # categorical distribution
        distribution = torch.distributions.Categorical(probabilities)
        predicted_tokens = distribution.sample()  # [bs, l, m]
        
        # mask: [bs, l, m] -> [bs,]
        num_masked = torch.sum(mask, dim=(1, 2))[0]

        # replace masked tokens with predicted tokens
        # predicted_tokens: [bs, l, m]
        predicted_tokens = torch.where(mask, predicted_tokens, masked_tokens)
        
        # confidence
        # gather([bs, l, m, 2 ** (k // m)], -1, [bs, l, m, 1]) -> [bs, l, m]
        confidence = torch.gather(probabilities, -1, predicted_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Ignore existing tokens by overwriting the confidence.
        confidence = torch.where(mask, confidence, torch.inf)

        # gumbel noise
        noise = gumbel.sample(predicted_tokens.size()) * randomize_temperature * (1 - progress)
        confidence = torch.log(confidence) + noise.to(device)

        # masking ratio
        mask_ratio = get_masking_ratio(progress, mode=mask_schedule_strategy).to(device)
        
        # min = 1, max = num_masked - 1
        mask_len = torch.floor(mask_ratio * num_maskable)
        num_tokens_to_mask = torch.clamp(mask_len, torch.ones_like(num_masked), num_masked-1).long()
        sorted_confidence = torch.sort(confidence.view(num_samples, -1), dim=-1).values
        threshold = sorted_confidence[:, num_tokens_to_mask - 1]

        should_mask = (confidence <= threshold.unsqueeze(-1).unsqueeze(-1))
        masked_tokens = torch.where(should_mask, mask_token, predicted_tokens)
        mask = (masked_tokens == mask_token)
        num_sampled += torch.where(should_mask, 0, 1)
        l_full_tokens.append(predicted_tokens.detach().cpu())
        
        # update masked_codes
        masked_codes = model.preprocess_tokens(masked_tokens)

    predicted_tokens = combine_factorized_tokens(predicted_tokens, codebook_size, codebook_splits)
    generated_image = vqgan_model.decode_tokens(predicted_tokens)
    return generated_image, l_full_tokens

@torch.no_grad()
def make_list_pred_ids_to_grid(
    pred_ids: List[Tensor],
    codebook_size: int, 
    codebook_splits: int, 
    vq_model: torch.nn.Module | object,
) -> Tensor:
    # pred_ids: list[Tensor[bs, l, m]]    
    pred_img_grid = []
    for p_ids in pred_ids:
        p_ids = p_ids.cuda()
        p_ids = combine_factorized_tokens(p_ids, codebook_size, codebook_splits)
        decoded_img = vq_model.decode_tokens(p_ids)
        pred_img_grid.append(decoded_img)
    
    return make_grid(pred_img_grid, nrow=1)
    

class MaskGitLFQTrainer:
    def __init__(self,
                 cfg: NameSpace,
                 logger: TensorboardLogger,
                 accelerator: accelerate.Accelerator,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 ):
        self.accelerator = accelerator
        self.cfg = cfg
        self.vae_cfg = cfg.vae_cfg
                
        # logger
        self.logger = logger
        
        # vae
        self.vae_encoder = Encoder(**cfg.vae_cfg.encoder_cfg)
        self.vae_decoder = Decoder(**cfg.vae_cfg.decoder_cfg)
        self.quantizer = LFQ(**cfg.vae_cfg.lfq_cfg)
        # load pretrained encoder, decoder, and quantizer
        self.load_encoder_decoder(self.vae_encoder,
                                  self.vae_decoder,
                                  self.quantizer)
        
        # move to device
        self.vae_encoder = self.vae_encoder.to(self.accelerator.device)
        self.vae_decoder = self.vae_decoder.to(self.accelerator.device)
        self.quantizer = self.quantizer.to(self.accelerator.device)
        
        # set eval mode
        self.vae_encoder.eval()
        self.vae_decoder.eval()
        self.quantizer.eval()
    
        # maskgit model
        self.maskgit_cfg = cfg.maskgit_cfg
        self.maskgit = MaskGit(**cfg.maskgit_cfg.model_cfg)
        self.masked_token = self.maskgit.mask_token
        self.split_m = cfg.maskgit_cfg.split_m
        
        # maskgit loss
        # self.maskgit_loss = MaskGitLoss(cfg.maskgit_cfg.loss_cfg)
        
        # dataloader
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # optimizer
        opt_cfg = cfg.optimizer_cfg
        self.opt_maskgit = get_optimizer(params=self.maskgit.parameters(), **opt_cfg.maskgit_opt)
        self.lr_scheduler_maskgit = get_scheduler(self.opt_maskgit, **opt_cfg.maskgit_sched)
        
        # prepare
        self.quantizer, self.maskgit, self.opt_maskgit, self.lr_scheduler_maskgit, self.train_dataloader, self.val_dataloader = \
            self.accelerator.prepare(self.quantizer, self.maskgit, self.opt_maskgit, self.lr_scheduler_maskgit,
                                     self.train_dataloader, self.val_dataloader)
        
        # EMA
        self.ema_maskgit = EMA(self.maskgit, beta=cfg.ema_cfg.beta, 
                               update_after_step=cfg.ema_cfg.update_after_step, 
                               update_every=cfg.ema_cfg.update_every,
                               state_include_online_model=True)
        self.accelerator.register_for_checkpointing(self.ema_maskgit)

        # steps counter
        self.steps_counter = StepsCounter()
        self.accelerator.register_for_checkpointing(self.steps_counter)
        
        # metrics
        self.metrics = AnalysisFusionAcc(test_metrics=MetricsByTask.ON_TRAIN)
        logger.info(f'[u][State][/u] metrics are calculated: {self.metrics.tested_metrics}')
        
        # cfg
        self.log_cfg = self.cfg.logger_config
        self.train_cfg = self.cfg.train_cfg
        self.save_checker_cfg = cfg.save_checker_cfg
        
        # aux loss
        self.use_aux_loss = cfg.maskgit_cfg.use_aux_loss
        if self.use_aux_loss:
            # says we often deal with RGB images
            self.aux_loss_module = get_loss('drmffusion', channel=3)
            warnings.warn('use auxiliary loss for training, gradient will be back-propagated to the decoder, \
                which maybe use more memory')
        
        # save checker
        self.save_checker = BestMetricSaveChecker(metric_name=self.save_checker_cfg.metric_name, 
                                                  check_order=self.save_checker_cfg.check_order)
        
        # resume training
        if self.train_cfg.resume_training_path is not None:
            self.resume(self.train_cfg.resume_training_path)
            
        # just load model
        if (self.train_cfg.pretrained_model_path is not None and 
            self.train_cfg.resume_training_path is None):
            self.load_from_ckpt(self.train_cfg.pretrained_model_path)
        
        # progress bar
        self.tbar, (self.train_iter_task, self.val_iter_task) = EasyProgress.easy_progress(
                                                                    ["Train Iter", "Validation Iter"], [len(train_dataloader), len(val_dataloader)],
                                                                    is_main_process=accelerator.is_main_process,
                                                                    tbar_kwargs={'console': logger.console},
                                                                    debug=cfg.debug)
        _, self.val_iter_task = EasyProgress.easy_progress(["Validation Iter"], [len(val_dataloader)],
                                                            is_main_process=accelerator.is_main_process,
                                                            tbar_kwargs={'console': logger.console},
                                                            debug=cfg.debug)
    
    def make_simple_vq_model_to_sample(self, token_size: list[int]):
        encoder = self.vae_encoder
        decoder = self.vae_decoder
        quantizer = self.quantizer
        
        class SimpleVQSampleModel(torch.nn.Module):
            def __init__(self, encoder, decoder, quantizer, token_size: list[int]):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.quantizer = quantizer
                self.token_size = token_size
                
            def forward(self, x: Tensor):
                return self.decode_tokens(x)
            
            def decode_tokens(self, predicted_indices: Tensor):
                decoded_codes = self.quantizer.indices_to_codes(predicted_indices)
                # to 2d size
                decoded_codes = rearrange(decoded_codes, 'b c (h w) -> b c h w',
                                          h=self.token_size[0], w=self.token_size[1]).float()
                decoded_img = self.decoder(decoded_codes)
                return decoded_img
        
        return SimpleVQSampleModel(encoder, decoder, quantizer, token_size)
    
    def maskgit_loss(self, logits: Tensor, target_ids: Tensor, masks: Tensor):
        # logits: [bs, l, m, 2 ** (k // m)]
        # target_ids: [bs, l * m]
        # masks: [bs, l * m]
        
        if logits.ndim == 4:
            # which is n_class at last dim
            logits_for_loss = rearrange(logits, 'b ... logits -> b (...) logits')[masks]
        elif logits.ndim == 3:
            # assume to be [bs, l * m, n_class]
            logits_for_loss = logits[masks]
        else:
            raise ValueError(f'Invalid logits dimension: {logits.ndim}')
        
        # accuracy
        acc = (logits_for_loss.argmax(dim=-1) == target_ids[masks]).float().mean()
        
        loss = F.cross_entropy(logits_for_loss, target_ids[masks],
                               label_smoothing=self.maskgit_cfg.label_smoothing)
        
        return loss, {'loss': loss, 'correct_tokens': acc}
        
    @property
    def train_n_iter(self):
        return self.steps_counter.n_train_steps
    
    @property
    def val_n_iter(self):
        return self.steps_counter.n_val_steps
    
    @property
    def may_update_progress(self):
        return self.accelerator.is_main_process and not self.cfg.debug
    
    def load_encoder_decoder(self, encoder: Encoder, decoder: Decoder, quantizer: LFQ):
        if self.vae_cfg.pretrained_ckpt.endswith('.ckpt'):  # is official checkpoint
            ckpt = torch.load(self.vae_cfg.pretrained_ckpt, weights_only=True)['state_dict']
            enc_params = {}
            dec_params = {}
            for k, v in ckpt.items():
                if k.startswith('encoder'):
                    enc_params[k.replace('encoder.', '')] = v
                elif k.startswith('decoder'):
                    dec_params[k.replace('decoder.', '')] = v
            encoder.load_state_dict(enc_params)
            decoder.load_state_dict(dec_params)
        else:  # is post-trained checkpoint
            ckpt = accelerate.utils.load_state_dict(self.vae_cfg.pretrained_ckpt)
            encoder.load_state_dict(ckpt['ema_encoder']['ema_model'])
            decoder.load_state_dict(ckpt['ema_decoder']['ema_model'])
            quantizer.load_state_dict(ckpt['ema_lfq']['ema_model']) if 'ema_lfq' in ckpt else None
        
        self.logger.info(f'[u][State][/u] load pretrained encoder, decoder, and quantizer from {self.vae_cfg.pretrained_ckpt}')

    def load_from_ckpt(self, ckpt_path: str, strict: bool = True):
        self.logger.info(f'[u][State][/u] load ckpt: {ckpt_path}')
        if ckpt_path.endswith('.pth'):
            ckpt = torch.load(ckpt_path)
        elif ckpt_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            ckpt = load_file(ckpt_path, device=self.accelerator.device)
        else:
            raise ValueError(f'Invalid ckpt format: {ckpt_path}')
        self.ema_maskgit.load_state_dict(ckpt['ema_maskgit'])
        self.steps_counter.load_state_dict(ckpt['steps_counter'])
        self.logger.info('[u][State][/u] load ckpt done')
        
    def state_dict(self):
        unwarp_maskgit = self.accelerator.unwrap_model(self.maskgit)
        ckpt = dict(
            mask_git=unwarp_maskgit.state_dict(),
            ema_maskgit=self.ema_maskgit.state_dict(),
            steps_counter=self.steps_counter.state_dict(),
        )
        return ckpt
    
    def resume(self, path: str):
        self.accelerator.load_state(path)
        self.logger.info(f'[b][State][/b] load training state from {path}')
        self.accelerator.wait_for_everyone()
        
    def save_state(self):
        self.accelerator.save_state(self.cfg.output_dir)
        self.logger.info(f'[g][State][/g] save training state at {self.cfg.output_dir} for global step {self.steps_counter.n_train_steps}')
        self.accelerator.wait_for_everyone()
        
    def save_ema(self):
        params = self.accelerator.unwrap_model(self.ema_maskgit).state_dict()
        self.accelerator.save(params, self.cfg.save_model_path)
        self.logger.info(f'[r][EMA][/r] save ema model at {self.cfg.save_model_path} for global step {self.steps_counter.n_train_steps}')
        self.accelerator.wait_for_everyone()
        
    def encode(self, x: Tensor):
        return self.vae_encoder(x)
    
    def encode_with_loss(self, x: Tensor, with_loss: bool=False):
        z = self.encode(x)
        lfq = self.quantizer
        if with_loss:
            (quant, indices, entropy_loss), loss_breakdown = lfq(z, return_loss_breakdown=True)
            return quant, indices, entropy_loss, loss_breakdown
        quant, indices, _ = lfq(z, return_loss_breakdown=False)
        return quant, indices
    
    def decode(self, quant: Tensor):
        with torch.no_grad():
            return self.vae_decoder(quant.float().cuda())
    
    def get_train_input(self) -> Generator[BatchedInput, None, None]:
        # get input from train dataloader
        """
        to guide fusion pair, i.e., [vis, ir] image pair and instruction text
        
        opt 1. instruction text are encoded by a pretrained model, e.g., T5 model
        opt 2. instruction text are encoded on the fly.
        
            image pair: [vis, ir], tuple of Tensor
        Return: 
            instruction: Tensor
        """
        _loader = iter(self.train_dataloader)
        while True:
            try:
                batch = next(_loader)
                yield make_batch_input(batch)
            except StopIteration:
                _loader = iter(self.train_dataloader)
                batch = next(_loader)
                yield make_batch_input(batch)
                
    def get_val_input(self) -> Generator[BatchedInput, None, None]:
        _loader = iter(self.val_dataloader)
        while True:
            batch = next(_loader)
            yield make_batch_input(batch)
            
    def mask_inp_ids(self, inp_ids: Tensor):
        # to codes
        inp_codes = self.quantizer.indices_to_bits(inp_ids) * 2 - 1
        
        return self.mask_inp_codes(inp_codes)
    
    # =========== mask the input codes after lfq ===========
    def mask_inp_codes(self, inp_codes: Tensor):
        bs, seq_len, codebook_k = inp_codes.size()
        device = inp_codes.device
        assert codebook_k % self.split_m == 0, f'codebook_k: {codebook_k} should be divisible by split_m: {self.split_m}'
        seq_len = int(np.ceil(seq_len * self.split_m))
        
        # mask some codes
        timesteps = torch.zeros((bs,), device=device).float().uniform_(0, 1.0)
        mask_ratio = torch.acos(timesteps) / (math.pi * 0.5) # arccos schedule
        mask_ratio = torch.clamp(mask_ratio, min=1e-6, max=1.)
        num_token_masked = (seq_len * mask_ratio).round().clamp(min=1)
        batch_randperm = torch.rand(bs, seq_len, device=device).argsort(dim=-1)
        masks = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        
        # factorize and mask codes
        factorized_codes = rearrange(inp_codes, 'b l (m c) -> b (l m) c', m=self.split_m)
        factorized_codes = factorized_codes * (1 - masks.float().unsqueeze(-1))
        factorized_codes = rearrange(factorized_codes, 'b (l m) c -> b l (m c)', m=self.split_m)
        
        # [b, l, k], [b, l * m]
        return factorized_codes, masks
    
    @torch.no_grad()
    def decode_recovered_tokens(self, logits: Tensor, img_size: tuple[int]=None):
        # logits from model(masked_tokens, instruction)
        # shape as (bs, n, m, 2 ** (k // m))
        
        # to (bs, n, m)
        logits = logits.argmax(dim=-1)
        indices = combine_factorized_tokens(logits, codebook_size=self.quantizer.codebook_size, splits=self.split_m)
        decoded_codes = self.quantizer.indices_to_codes(indices)
        size = img_size or self.maskgit_cfg.sample_size
        assert size is not None, 'sample_size should be specified'
        decoded_codes = rearrange(decoded_codes, 'b c (h w) -> b c h w', h=size[0], w=size[1])
        decoded_img = self.decode(decoded_codes)
        decoded_img = (decoded_img + 1) / 2  # to [0, 1]
        
        return decoded_img
    
    def auxiliary_loss(self, logits: Tensor, source1: Tensor, source2: Tensor) -> dict:
        # decode back to pixel space
        pixel_img = self.decode_recovered_tokens(logits)
        return self.aux_loss_module(pixel_img, source1, source2)
    
    def get_factorized_ids(self, codes: Tensor):
        # gt codes: (bs, l, k)
        # return ids: (bs, l, m)
        gt_codes_factorized = rearrange(codes, 'b l (m c) -> b l m c', m=self.split_m)
        codebook_dim_factorized = self.quantizer.codebook_dim // self.split_m
        gt_indices_lst = []
        for i in range(self.split_m):
            part_codes = gt_codes_factorized[:, :, i]  # [b, l, c]
            part_codes = (part_codes + 1) / 2  # to [0, 1]
            indices = 2 ** torch.arange(0, codebook_dim_factorized, 1, 
                                        dtype=torch.long, device=codes.device)
            indices = (part_codes * indices).sum(-1)  # [b, l]
            gt_indices_lst.append(indices.long())
        # [b, l, m] -> [b, l * m]
        return torch.stack(gt_indices_lst, dim=-1).view(codes.size(0), -1)
    
    def train_step(self, batch: BatchedInput):
        instruction = batch.instruction
        source1 = batch.source1
        source2 = batch.source2
        
        if batch.image_codes is not None:
            inp_codes = batch.image_codes
            source1_codes = batch.source1_codes
            source2_codes = batch.source2_codes
        else:
            # to codes
            inp_codes = self.encode_with_loss(batch.image)[0]
            source1_codes = self.encode_with_loss(batch.source1)[0]
            source2_codes = self.encode_with_loss(batch.source2)[0]

        # mask images
        # masked_codes: [b, l, c], masks: [b, l]
        masked_codes, masks = self.mask_inp_codes(inp_codes)

        with self.accelerator.accumulate() and self.accelerator.autocast():
            # to maskgit
            # shape as (bs, n, m, 2 ** (k // m))
            logits = self.maskgit(masked_codes, (source1_codes, source2_codes), instruction)
            
            # loss
            target_ids = self.get_factorized_ids(inp_codes)  # [bs, n * m]
            loss, mask_loss = self.maskgit_loss(logits, target_ids, masks)
            if self.train_cfg.use_auxiliary_loss:
                aux_loss = self.auxiliary_loss(logits, source1, source2)
                loss += aux_loss['loss']
                
            # backward loss
            self.opt_maskgit.zero_grad()
            self.accelerator.backward(loss)
            self.opt_maskgit.step()
            
        # update ema
        if self.accelerator.sync_gradients:
            self.ema_maskgit.update()
            self.lr_scheduler_maskgit.step()
        
        self.steps_counter.update('train')
        
        # udpate progress bar
        tbar = self.tbar
        train_iter_task = self.train_iter_task
        if self.may_update_progress:
            tbar.update(train_iter_task, total=self.train_cfg.max_train_steps, completed=self.train_n_iter, visible=True,
                        description=f'Train Iter [{self.train_n_iter}/{self.train_cfg.max_train_steps}] - ' + \
                                    f'mask_loss: {mask_loss["loss"].item():.4f}, ' + \
                                    f'correct_tokens: {mask_loss["correct_tokens"].item():.4f}, ' + \
                                    f'aux_loss: {round(aux_loss["loss"].item(), 4) if self.train_cfg.use_auxiliary_loss else "N/A"}')
        
        # log
        if self.train_n_iter % self.log_cfg.log_every == 0:
            self.logger.info(f'[b][Train][/b] global step: {self.train_n_iter} | lr: {self.lr_scheduler_maskgit.get_last_lr()[0]:.4e} | ' + \
                             f'mask_loss: {mask_loss["loss"].item():.4f} | correct_tokens: {mask_loss["correct_tokens"].item():.4f} | ' + \
                             f'aux_loss: {round(aux_loss["loss"].item(), 4) if self.train_cfg.use_auxiliary_loss else "N/A"}')
            self.logger.log_curves({'loss/mask_loss': mask_loss["loss"].item(), 
                                    'acc/correct_tokens': mask_loss["correct_tokens"].item(),
                                    'loss/aux_loss': aux_loss["loss"].item() if self.train_cfg.use_auxiliary_loss else 0.0}, 
                                    self.train_n_iter)
            
        # log train image
        if self.train_n_iter % self.log_cfg.log_train_img_every == 0:
            self.logger.log_image(batch.source1_clean[0], 'train_img/source1', self.train_n_iter)
            self.logger.log_image(batch.source2_clean[0], 'train_img/source2', self.train_n_iter)
            # umasked image, only one image
            self.logger.log_image(self.decode_recovered_tokens(logits[0:1]).squeeze(0), 'train_img/pred', self.train_n_iter)
    
    @torch.no_grad()
    def val_step(self, batch: BatchedInput):
        accelerator = self.accelerator
        
        # get data
        instruction = batch.instruction
        source1_codes = batch.source1_codes
        source2_codes = batch.source2_codes
        n_samples = source1_codes.size(0)
        
        # sample
        # none_label = None
        gen_img, sampled_full_tokens = sample(
            model=self.ema_maskgit.ema_model,
            vqgan_model=self.make_simple_vq_model_to_sample(token_size=self.maskgit_cfg.sample_size),
            num_samples=n_samples,
            # labels=none_label,  # no label here, conditions only in source1_codes, source2_codes, and instruction
            conditions=((source1_codes, source2_codes), instruction),
            num_steps=self.maskgit_cfg.num_steps,
            mask_schedule_strategy=self.maskgit_cfg.mask_schedule_strategy,
            mask_token=self.masked_token,
            sample_size=self.maskgit_cfg.sample_size,
            codebook_size=self.quantizer.codebook_size,
            codebook_splits=self.split_m,
            # guidance_scale=self.maskgit_cfg.guidance_scale,
        )
        
        return gen_img, sampled_full_tokens
    
    def train_loop(self):
        logger = self.logger
        logger.info('[g][Train][/g] Start training ...')
        logger.info(f'[g][Train][/g] Training steps per epoch: {len(self.train_dataloader)}')
        logger.info(f'[g][Train][/g] Validation steps per epoch: {len(self.val_dataloader)}')
        logger.info(f'[g][Train][/g] Save state every {self.train_cfg.save_state_every} steps')
        logger.info(f'[g][Train][/g] Validation every {self.train_cfg.val_every} steps')
        
        if self.train_cfg.sanity_check_val:
            self.val_loop()
        
        for batch in self.get_train_input():
            # train step
            self.train_step(batch)
            
            # save ckpt
            if self.train_n_iter % self.train_cfg.save_state_every == 0:
                self.save_state()
                
            # validation
            if self.train_n_iter % self.train_cfg.val_every == 0:
                # close training tbar temporarily
                self.tbar.update(self.train_iter_task, visible=False)
                
                # validation loop
                mean_metrics = self.val_loop()
            
                # save ema
                save_or_not = self.save_checker(mean_metrics)
                if save_or_not:
                    self.save_ema()
                    
            # end training
            if self.train_n_iter >= self.train_cfg.max_train_steps:
                logger.info(f'[g][Train][/g] Training steps reached max steps: {self.train_cfg.max_train_steps}, end training')
                break
    
    @torch.no_grad()
    def val_loop(self):
        accelerator = self.accelerator
        logger = self.logger
        tbar = self.tbar
        
        logger.info('[b][Validation][/b] Start validation ...')
        log_samples = []
        log_sources = []
        
        self.maskgit.eval()
        self.ema_maskgit.ema_model.eval()
        
        for n_iter, batch in enumerate(self.get_val_input(), start=1):
            gen_img, sampled_full_ids = self.val_step(batch)
            
            gen_img_metric = (gen_img + 1) / 2
            source1 = batch.source1_clean  # assume it on (0, 1)
            source2 = batch.source2_clean  # assume it on (0, 1)
            
            assert source1 is not None and source2 is not None, 'source1 and source2 should not be None'
            self.metrics(
                (source1, source2),  # e.g., (vis, ir)
                gen_img_metric,
            )
            
            # update progress bar and counter
            # tbar.update(
            #     self.val_iter_task, 
            #     total=len(self.val_dataloader), 
            #     completed=n_iter, 
            #     visible=True,
            # )
            logger.info(f'[b][Validation][/b] global step: {self.val_n_iter}')
            self.steps_counter.update('val')
            
            if self.log_cfg.log_n_samples > len(log_samples):
                logger.info(f'log {len(log_samples)} samples')
                log_samples.append(gen_img)
                log_sources.append((source1, source2))
                
            if n_iter > 2:
                logger.info('break validation for debug')
                break
            
        # get distributed metrics and images
        process_metric = self.metrics.acc_ave
        dist_mean_metric = self.distributed_mean_dict(process_metric)
        
        # log
        result_metric_str = self.metrics.result_str(dist_mean_metric)
        logger.info(f'[b][Validation][/b] global step: {self.val_n_iter}, metrics are: \n \
                    {result_metric_str}')
        
        # grid sampled images
        if self.log_cfg.log_val_traj:
            vq_model = self.make_simple_vq_model_to_sample(token_size=self.maskgit_cfg.sample_size)
            dist_sampled_img_grid = make_list_pred_ids_to_grid(sampled_full_ids, self.quantizer.codebook_size, 
                                                               self.split_m, vq_model)
            self.logger.log_image(dist_sampled_img_grid, 'val_sampled_images', self.val_n_iter)
        
        # log curves and images
        self.logger.log_curves(dist_mean_metric, self.val_n_iter, prefix='metrics')
        
        # image grid
        dist_log_samples = self.accelerator.gather(torch.cat(log_samples, dim=0))
        sources_1, sources_2 = list(zip(*log_sources))
        dist_s1, dist_s2 = self.accelerator.gather(torch.cat(sources_1, dim=0)), self.accelerator.gather(torch.cat(sources_2, dim=0))
        img_grid = make_grid(dist_log_samples, nrow=4)
        s1_grid, s2_grid = make_grid(dist_s1, nrow=4), make_grid(dist_s2, nrow=4)
        self.logger.log_image((img_grid + 1) / 2, 'val_samples', self.val_n_iter)
        self.logger.log_image(s1_grid, 'val_sources_1', self.val_n_iter)
        self.logger.log_image(s2_grid, 'val_sources_2', self.val_n_iter)
        self.logger.info('[b][Validation][/b] log curves and images done')
        
        # set to train mode
        self.maskgit.train()
        
        return dist_mean_metric
        
    def distributed_mean_dict(self, d: dict):
        assert isinstance(d, dict), f'Input should be a dict, but {type(d)} is given'
        
        if self.accelerator.num_processes == 1:
            return d
        
        dist_d_lst = [None for _ in range(self.accelerator.num_processes)]
        self.accelerator.wait_for_everyone()
        torch.distributed.all_gather_object(dist_d_lst, d)
        
        mean_d = {}
        for k in d.keys():
            mean_d[k] = sum([di[k].item() if isinstance(di[k], torch.Tensor) else di[k] 
                             for di in dist_d_lst]) / len(dist_d_lst)
            
        return mean_d
    

## helper functions

def experimental_logger(exp_name: str='VQ_LFQ', config_args: NameSpace=None):
    state = PartialState()
    if state.is_main_process:        
        # unify tensorboard logger and file logger
        # will save config file to log dir
        logger = TensorboardLogger(
            args=config_args,
            tsb_comment=config_args.comment,
            file_stream_log=True,
            method_dataset_as_prepos=True,
        )
    else:
        logger = NoneLogger(cfg=config_args, name=exp_name)
    
    return logger
    
def specify_weights_path(args: NameSpace, logger, accelerator: accelerate.Accelerator):
    if accelerator.is_main_process:
        weight_path = osp.join(logger.log_file_dir, 'weights')
        if accelerator.is_main_process:
            os.makedirs(weight_path, exist_ok=True)
        args.output_dir = weight_path
        args.save_model_path = osp.join(args.output_dir, 'ema_model.pth')
    else:
        # for ddp broadcast
        args.output_dir = args.save_model_path = args.save_base_path = [None]
    
    if accelerator.use_distributed:
        (args.output_dir, 
         args.save_model_path,
         args.save_base_path) = accelerate.utils.broadcast_object_list([args.output_dir, 
                                                                        args.save_model_path, 
                                                                        args.save_base_path])
    accelerator.project_configuration.project_dir = args.output_dir
    logger.info(f'model weights will be save at {args.output_dir}')
    
    return args

def main(args: NameSpace):
    accelerator = accelerate.Accelerator(mixed_precision='no',
                                         dataloader_config=DataLoaderConfiguration(split_batches=False,
                                                                                   even_batches=False,
                                                                                   non_blocking=True,
                                                                                   dispatch_batches=None,),
                                         project_config=ProjectConfiguration(project_dir='log_file/',
                                                                             total_limit=args.ckpt_max_limit,
                                                                             automatic_checkpoint_naming=True,),
                                         gradient_accumulation_plugin=GradientAccumulationPlugin(adjust_scheduler=False,
                                                                                                 sync_with_dataloader=False,
                                                                                                 sync_each_batch=False,),
                                         )
    
    # get logger
    logger = experimental_logger(exp_name=args.dataset_name, 
                                 config_args=args)
    
    # specify weights path
    args = specify_weights_path(args, logger, accelerator)
    
    # dataset
    (train_ds, train_dl, val_ds, val_dl), args = get_train_dataset(args, 
                                                                   init_with_default_ds_cfg=False)
    
    # init model
    logger.info('Init MaskGiT trainer to instruct image fuison: {}'.format(args.dataset_name))
    engine = MaskGitLFQTrainer(args, logger, accelerator, train_dl, val_dl)
    
    # train it
    engine.train_loop()
    
    # end of training
    accelerator.end_training()    
    
    
if __name__ == "__main__":
    import sys
    import argparse
    from utils.log_utils import LoguruLogger
    logger = LoguruLogger.logger(sink=sys.stdout)
    LoguruLogger.add('log_file/running_traceback.log', format="{time:MM-DD hh:mm:ss} {level} {message}", 
                    level="WARNING", backtrace=True, diagnose=True, mode='w')
    LoguruLogger.add(sys.stderr, format="{time:MM-DD hh:mm:ss} {level} {message}", 
                    level="ERROR", backtrace=True, diagnose=True)
    
    ## training arguments
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="maskgit")
        parser.add_argument("--dataset_name", type=str, default='dif_msrs')
        parser.add_argument("--ckpt_max_limit", type=int, default=10)
        parser.add_argument("--comment", type=str, default='')
        parser.add_argument("--debug", action="store_true")
        
        args = parser.parse_args()
        
        # used for logger, not used in training
        args.dataset = args.dataset_name
        args.full_arch = 'maskgit'
        
        return args
    
    ## main
    args = parse_args()
    model_cfg = NameSpace.init_from_yaml(args.config)
    args = NameSpace.merge_parser_args(args, model_cfg)
    
    state = PartialState()    
    try:
        main(args)
    except Exception as e:
        if state.is_main_process:
            EasyProgress.close_all_tasks()
            logger.error('An Error Occurred! Please check the stacks in log_file/running_traceback.log')
            logger.exception(e)
        else:
            logger.exception(e)
        raise RuntimeError(f'Process {os.getpid()} encountered an error: {e}')
    
    