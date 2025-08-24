import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch import Tensor
import torch.distributed
from torch.utils.data import DataLoader
import accelerate
from accelerate import PartialState
from accelerate.utils import DataLoaderConfiguration, ProjectConfiguration, GradientAccumulationPlugin
from accelerate import DistributedDataParallelKwargs
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import argparse
from random import uniform
import os
import os.path as osp
import gc

from model.lfq_vae.autoencoder import Decoder, Encoder
# from model.lfq_vae.flux_autoencoder import Decoder, Encoder
from model.lfq_vae.quantizer import LFQ, LFQ_v0, BSQ, RLFQ
from model.lfq_vae.gan_loss.loss import VQLPIPSWithDiscriminator

from utils import (
    NameSpace,
    NoneLogger,
    TensorboardLogger,
    config_load,
    get_optimizer,
    get_scheduler,
    EMA,
    catch_any_error,
    EasyProgress,
    default,
    dict_to_str,
    BestMetricSaveChecker,
)
from utils.train_test_utils import get_train_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="lfq_vae_config.yaml")
    parser.add_argument("--exp_name", type=str, default='lfq_vae')
    parser.add_argument("--ckpt_max_limit", type=int, default=10)
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset", type=str, default='unify_image_fusion_vae')
    parser.add_argument("--resume_ckpt", type=str, default=None)
    
    args = parser.parse_args()
    return args

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
        
def get_collate_fn(key=['vi', 'ir']):
    def collate_fn(batch):
        output_batch = []
        for b in batch:
            output_batch.append([b[k] for k in key])
        for i in range(len(output_batch)):
            output_batch[i] = torch.stack(output_batch[i])
        return output_batch
    
    return collate_fn

class LFQ_VQ_VAE_Trainer:
    def __init__(self,
                 cfg: NameSpace,
                 logger: TensorboardLogger,
                 accelerator: accelerate.Accelerator,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader | None=None,
                 compile_model: bool=False,
                 ):
        self.cfg = cfg
        self.logger = logger
        self.log_cfg = cfg.logger_config

        # init models
        logger.info(f'[u][Model][/u] construct model ...')
        self.encoder = Encoder(**cfg.vae_cfg.encoder_cfg)
        self.decoder = Decoder(**cfg.vae_cfg.decoder_cfg)
        self.lfq = LFQ_v0(**cfg.vae_cfg.lfq_cfg)
        self.vq_loss_fn = VQLPIPSWithDiscriminator(**cfg.vae_cfg.vq_loss_cfg)
        self.encoder = torch.compile(self.encoder) if compile_model else self.encoder
        self.decoder = torch.compile(self.decoder) if compile_model else self.decoder
        
        # load pretrained weights
        _resume_state_or_ckpt = osp.isdir(cfg.resume_ckpt) and osp.exists(cfg.resume_ckpt)
        if cfg.vae_cfg.pretrained_ckpt is not None and cfg.resume_ckpt is None:
            self.load_from_ckpt(cfg.vae_cfg.pretrained_ckpt, is_pretrained=True)
        elif not _resume_state_or_ckpt:
            logger.info(f'[u][Weights][/u] load ckpt: {cfg.resume_ckpt}')
            self.load_from_ckpt(cfg.resume_ckpt, is_pretrained=False, load_ema=True)
        
        # augmentations
        assert hasattr(train_dataloader, '_augmentations_pipes'), 'train_dataloader must have attribute `_augmentations_pipes`'
        # assert hasattr(val_dataloader, '_augmentations_pipes'), 'val_dataloader must have attribute `_augmentations_pipes`'
        self.train_augmentations_pipes = train_dataloader._augmentations_pipes
        # self.val_augmentations_pipes = val_dataloader._augmentations_pipes
        
        # optimizer and lr scheduler
        logger.info(f'[u][Optimizer][/u] construct optimizer ...')
        opt_cfg = cfg.optimizer_cfg
        gan_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.lfq.parameters())
        self.opt_enc_dec = get_optimizer(params=gan_params, **opt_cfg.gan_opt)
        self.opt_disc = get_optimizer(params=self.vq_loss_fn.discriminator.parameters(), **opt_cfg.disc_opt)
        self.lr_scheduler_enc_dec = get_scheduler(self.opt_enc_dec, **opt_cfg.gan_lr_sched)
        self.lr_scheduler_disc = get_scheduler(self.opt_disc, **opt_cfg.disc_lr_sched)
        
        # prepare
        self.accelerator = accelerator
        self.prepare_to_train()
        (self.encoder, self.decoder, self.lfq, self.vq_loss_fn.discriminator, 
         self.opt_enc_dec, self.opt_disc, self.train_dataloader, self.val_dataloader) = \
            self.accelerator.prepare(self.encoder, self.decoder, self.lfq, self.vq_loss_fn.discriminator, 
                                     self.opt_enc_dec, self.opt_disc, train_dataloader, val_dataloader)
        self.accelerator.register_for_checkpointing(self.lr_scheduler_enc_dec, self.lr_scheduler_disc)
        
        # EMA model
        ema_cfg = cfg.ema_cfg
        self.ema_encoder = EMA(self.encoder, beta=ema_cfg.beta, update_after_step=ema_cfg.update_after_step, 
                             update_every=ema_cfg.update_every, state_include_online_model=True)
        self.ema_decoder = EMA(self.decoder, beta=ema_cfg.beta, update_after_step=ema_cfg.update_after_step, 
                             update_every=ema_cfg.update_every, state_include_online_model=True)
        self.ema_lfq = EMA(self.lfq, beta=ema_cfg.beta, update_after_step=ema_cfg.update_after_step, 
                             update_every=ema_cfg.update_every, state_include_online_model=True)
        self.accelerator.register_for_checkpointing(self.ema_encoder, self.ema_decoder, self.ema_lfq)
            
        # training states
        self.train_state = StepsCounter()
        self.accelerator.register_for_checkpointing(self.train_state)
        
        # resume training states
        if cfg.resume_ckpt is not None and _resume_state_or_ckpt:
            self.resume(cfg.resume_ckpt)
        
        # metrics
        self._psnr_fn = PeakSignalNoiseRatio(data_range=1.).to(accelerator.device)
        self._ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.).to(accelerator.device)
        
        # train configs
        self.train_cfg = cfg.train_cfg
        
        # save checker
        self.save_checker_cfg = cfg.save_checker_cfg
        self.save_checker = BestMetricSaveChecker(metric_name=self.save_checker_cfg.metric_name,
                                                  check_order=self.save_checker_cfg.check_order)
        
        # progress bar
        self.tbar, (self.train_iter_task, self.val_iter_task) = EasyProgress.easy_progress(
                                                                    ["Train Iter", "Validation Iter"], [len(train_dataloader), len(val_dataloader)],
                                                                    is_main_process=accelerator.is_main_process,
                                                                    tbar_kwargs={'console': logger.console},
                                                                    debug=args.debug)

    def prepare_to_train(self):
        self.encoder.train()
        self.decoder.train()
        self.lfq.train()
        self.vq_loss_fn.discriminator.train()
        
        self.encoder.requires_grad_(True)
        self.decoder.requires_grad_(True)
        self.lfq.requires_grad_(True)
        self.vq_loss_fn.discriminator.requires_grad_(True)
        
        # discriminator may have batch norm layer
        self.vq_loss_fn.discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(self.vq_loss_fn.discriminator)
        self.logger.info('[u][Model][/u] convert discriminator to sync batch norm')
    
    @property
    def train_n_iter(self):
        return self.train_state.n_train_steps + 1  # start from 1
    
    @property
    def val_n_iter(self):
        return self.train_state.n_val_steps
    
    def load_from_ckpt(self, ckpt_path: str, strict: bool = True, *, is_pretrained: bool=False, load_ema: bool=False):
        from packaging import version
        
        self.logger.info('[u][Weights][/u] load previous training ckpt: {}'.format(ckpt_path))
        _pt_weights_only = version.parse(torch.__version__) >= version.parse('2.0.0')
        
        if ckpt_path.endswith('.pth'):
            if _pt_weights_only:
                ckpt = torch.load(ckpt_path, weights_only=True)
            else:
                ckpt = torch.load(ckpt_path)
        elif ckpt_path.endswith('.ckpt'):  # ckpt to suit the pretrained file
            if _pt_weights_only:
                ckpt = torch.load(ckpt_path, weights_only=True)
            else:
                ckpt = torch.load(ckpt_path)
            ckpt = ckpt['state_dict']
        elif ckpt_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            ckpt = load_file(ckpt_path, device=self.accelerator.device)
        else:
            raise ValueError(f'Invalid ckpt format: {ckpt_path}')
        
        # is pretrained lfq vae
        if is_pretrained:
            # handle the keys
            enc_params = {}
            dec_params = {}
            lfq_params = None
            for k, v in ckpt.items():
                if k.startswith('encoder'):
                    enc_params[k.replace('encoder.', '')] = v
                elif k.startswith('decoder'):
                    dec_params[k.replace('decoder.', '')] = v
        else:
            if load_ema:
                enc_params = ckpt["ema_encoder"]["ema_model"]
                dec_params = ckpt["ema_decoder"]["ema_model"]
                lfq_params = ckpt["ema_lfq"]["ema_model"] if 'ema_lfq' in ckpt else None
            else:
                enc_params = ckpt["ema_encoder"]["online_model"]
                dec_params = ckpt["ema_decoder"]["online_model"]
                lfq_params = ckpt["ema_lfq"]["online_model"] if 'lfq' in ckpt else None
                
        self.encoder.load_state_dict(enc_params, strict=strict)
        self.decoder.load_state_dict(dec_params, strict=strict)
        if lfq_params is not None:
            self.lfq.load_state_dict(lfq_params, strict=strict)
            
        if "opt_enc_dec" in ckpt:
            self.opt_enc_dec.load_state_dict(ckpt["opt_enc_dec"], strict=strict)
        if "opt_disc" in ckpt:
            self.opt_disc.load_state_dict(ckpt["opt_disc"], strict=strict)
        if "lr_sched_enc_dec" in ckpt:
            self.lr_scheduler_enc_dec.load_state_dict(ckpt["lr_sched_enc_dec"], strict=strict)
        if "lr_sched_disc" in ckpt:
            self.lr_scheduler_disc.load_state_dict(ckpt["lr_sched_disc"], strict=strict)
            
        self.logger.info('[u][Weights][/u] load ckpt done')
        
    def state_dict(self, with_opt=True, with_lr_sched=True):
        unwarp_encoder = self.accelerator.unwrap_model(self.encoder)
        unwarp_decoder = self.accelerator.unwrap_model(self.decoder)
        unwarp_lfq = self.accelerator.unwrap_model(self.lfq)
        unwarp_discriminator = self.accelerator.unwrap_model(self.vq_loss_fn.discriminator)
        ckpt = dict(
            encoder=unwarp_encoder.state_dict(),
            decoder=unwarp_decoder.state_dict(),
            lfq=unwarp_lfq.state_dict(),
            discriminator=unwarp_discriminator.state_dict(),
            ema_encoder=self.ema_encoder.state_dict(),
            ema_decoder=self.ema_decoder.state_dict(),
            ema_lfq=self.ema_lfq.state_dict(),
            train_states=self.train_state.state_dict(),
        )
        if with_opt:
            ckpt["opt_enc_dec"] = self.opt_enc_dec.state_dict()
            ckpt["opt_disc"] = self.opt_disc.state_dict()
        if with_lr_sched:
            ckpt["lr_sched_enc_dec"] = self.lr_scheduler_enc_dec.state_dict()
            ckpt["lr_sched_disc"] = self.lr_scheduler_disc.state_dict()
            
        return ckpt
    
    def encode(self, x: Tensor, use_ema: bool=False):
        with self.accelerator.autocast():
            encoder = self.ema_encoder if use_ema else self.encoder
            if use_ema:
                encoder.eval()
            return encoder(x)
    
    def encode_with_loss(self, x: Tensor, use_ema: bool=False):
        z = self.encode(x, use_ema)
        lfq = self.ema_lfq if use_ema else self.lfq
        (quant, indices, codebook_loss), loss_breakdown = lfq(z, return_loss_breakdown=True)
        return quant, codebook_loss, indices, loss_breakdown
    
    def decode(self, quant: Tensor, use_ema: bool=False):
        with self.accelerator.autocast():
            decoder = self.ema_decoder if use_ema else self.decoder
            if use_ema:
                decoder.eval()
            return decoder(quant)
    
    def resume(self, path: str):
        self.logger.info(f'[b][State][/b] load training state from {path}')
        self.accelerator.load_state(path)
        self.accelerator.wait_for_everyone()
    
    def save_state(self):
        self.accelerator.wait_for_everyone()
        # to avoid InvalidHeaderDeserialization error, make ``safe_serialization=False``
        # see https://github.com/huggingface/transformers/issues/27397
        self.accelerator.save_state(self.cfg.output_dir, safe_serialization=False)
        self.logger.info(f'[g][State][/g] save training state at {self.cfg.output_dir} for global step {self.train_state.n_train_steps}')
        self.accelerator.wait_for_everyone()
        
    def save_ema(self):
        self.accelerator.wait_for_everyone()
        encoder_params = self.accelerator.unwrap_model(self.ema_encoder).state_dict()
        decoder_params = self.accelerator.unwrap_model(self.ema_decoder).state_dict()
        lfq_params = self.accelerator.unwrap_model(self.ema_lfq).state_dict()
        ema_ckpt = dict(
            ema_encoder=encoder_params,
            ema_decoder=decoder_params,
            ema_lfq=lfq_params,
        )
        self.accelerator.save(ema_ckpt, self.cfg.save_model_path, safe_serialization=False)
        self.logger.info(f'[r][EMA][/r] save ema model at {self.cfg.save_model_path} for global step {self.train_state.n_train_steps}')
        self.accelerator.wait_for_everyone()
        
    def metrics(self, x: Tensor, x_recon: Tensor):
        x = self.to_zero_one(x)
        x_recon = self.to_zero_one(x_recon)
        
        psnr = self._psnr_fn(x_recon, x).item()
        ssim = self._ssim_fn(x_recon, x).item()
        
        # self.logger.info(f'[g][Metrics][/g] PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')
        # self.logger.log_curve(psnr, 'psnr', self.train_n_iter)
        # self.logger.log_curve(ssim, 'ssim', self.train_n_iter)
        return {"psnr": psnr, "ssim": ssim}
    
    def forward(self, x: Tensor, use_ema: bool=False):
        quant, codebook_loss, _, loss_breakdown = self.encode_with_loss(x, use_ema)
        x_recon = self.decode(quant, use_ema)
        return x_recon, codebook_loss, loss_breakdown
    
    def __call__(self, x: Tensor, use_ema: bool=False):
        return self.forward(x, use_ema)
    
    def get_train_input(self):
        _loader = iter(self.train_dataloader)
        _basic_aug_prob = 1.0 # self.cfg.vae.aug_prob
        while True:
            try:
                batch = next(_loader)[0]
                if uniform(0, 1) < _basic_aug_prob * (1 - self.train_n_iter / self.train_cfg.max_train_steps):
                    batch = self.train_augmentations_pipes(batch)
                batch = self.to_neg_one_one(batch)
                yield batch  # has vi and ir in one batch, we unpack it and forward both
            except StopIteration:
                _loader = iter(self.train_dataloader)
                batch = next(_loader)[0]                
                if uniform(0, 1) < _basic_aug_prob * (1 - self.train_n_iter / self.train_cfg.max_train_steps):
                    batch = self.train_augmentations_pipes(batch)
                batch = self.to_neg_one_one(batch)
                yield batch
    
    def to_neg_one_one(self, x: Tensor):
        return x.clamp(0, 1) * 2 - 1
    
    def to_zero_one(self, x: Tensor):
        return ((x + 1) / 2).clamp(0, 1)
        

    def _train_step_enc_dec(self, 
                            x: Tensor,
                            x_recon: Tensor,
                            codebook_loss,
                            loss_breakdown):
        vq_loss, log_vq = self.vq_loss_fn(
            codebook_loss, loss_breakdown, x, x_recon, 0, self.train_n_iter,
            last_layer=self.get_last_layer(), split="train"
        )
                    
        self.opt_enc_dec.zero_grad()
        self.accelerator.backward(vq_loss)
        self.opt_enc_dec.step()
        # ema and lr scheduler update
        if self.accelerator.sync_gradients:
            self.ema_encoder.update()
            self.ema_decoder.update()
            self.ema_lfq.update()
            self.lr_scheduler_enc_dec.step()
        
        return vq_loss, log_vq
    
    def _train_step_disc(self, 
                         x: Tensor,
                         x_recon: Tensor,
                         codebook_loss,
                         loss_breakdown):
        disc_loss, log_disc = self.vq_loss_fn(
            codebook_loss, loss_breakdown, x, x_recon, 1, self.train_n_iter,
            last_layer=self.get_last_layer(), split="train"
        )
        
        self.opt_disc.zero_grad()
        self.accelerator.backward(disc_loss)
        self.opt_disc.step()
        if self.accelerator.sync_gradients:
            self.lr_scheduler_disc.step()
        
        return disc_loss, log_disc
    
    def train_step(self, x: Tensor):
        accelerator = self.accelerator
        
        vq_loss = disc_loss = log_vq = log_disc = None
        
        # accumulate gradient
        with accelerator.accumulate():
            x_recon, codebook_loss, loss_breakdown = self(x, use_ema=False)
            # encoder/decoder/lfq optimizer step
            vq_loss, log_vq = self._train_step_enc_dec(x, x_recon, codebook_loss, loss_breakdown)
            # discriminator optimizer step
            disc_loss, log_disc = self._train_step_disc(x, x_recon, codebook_loss, loss_breakdown)
            
        # log
        if self.train_n_iter % self.log_cfg.log_every == 0:
            logger = self.logger
            logger.log_curves(log_vq, self.train_n_iter)
            logger.log_curves(log_disc, self.train_n_iter)
            logger.log_curve(self.lr_scheduler_enc_dec.get_last_lr()[0], 'lr', self.train_n_iter)
            logger.info('Train iter {} - lr: {:.4e}, vq_loss: {:.4f}, disc_loss: {:.4f}, recon_l: {:.4f}, percep_l: {:.4f}'.format(self.train_n_iter, 
                                                                                                self.lr_scheduler_enc_dec.get_last_lr()[0], 
                                                                                                vq_loss, disc_loss, 
                                                                                                log_vq['train/reconstruct_loss'], 
                                                                                                log_vq['train/perceptual_loss']))

        # update training state
        self.train_state.update(mode='train')
        
        # progress bar
        tbar = self.tbar
        train_iter_task = self.train_iter_task
        if accelerator.is_main_process and not self.cfg.debug:
            tbar.update(train_iter_task, total=len(self.train_dataloader), completed=self.train_n_iter, visible=True,
                        description=f'Train Iter [{self.train_n_iter}/{len(self.train_dataloader)}] ' + \
                                    f'- g_loss: {log_vq["train/g_loss"]:.4f}, d_loss: {log_disc["train/disc_loss"]:.4f} ' + \
                                    f'- recon_l: {log_vq["train/reconstruct_loss"]:.4f}, percep_l: {log_vq["train/perceptual_loss"]:.4f}')
            
        return x_recon
        
    def val_step(self, x: Tensor):
        accelerator = self.accelerator
        
        with accelerator.autocast():
            x_recon, codebook_loss, loss_breakdown = self(x, use_ema=True)

        # metrics
        x_recon = x_recon.clamp(-1, 1)
        metrics = self.metrics(x, x_recon)
        
        # loss
        vq_loss, log_vq = self.vq_loss_fn(codebook_loss, loss_breakdown, x, x_recon, 0, self.train_n_iter,
                                            last_layer=self.get_last_layer(), split="val")
        disc_loss, log_disc = self.vq_loss_fn(codebook_loss, loss_breakdown, x, x_recon, 1, self.train_n_iter,
                                            last_layer=self.get_last_layer(), split="val")
        
        # progress bar
        tbar = self.tbar
        val_iter_task = self.val_iter_task
        if accelerator.is_main_process and not self.cfg.debug:
            # tbar.reset(val_iter_task)
            tbar.update(val_iter_task, total=len(self.val_dataloader), completed=self.val_n_iter, visible=True,
                        description=f'Validation Iter [{self.val_n_iter}/{len(self.val_dataloader)}] - PSNR: {metrics["psnr"]}, SSIM: {metrics["ssim"]}')

        return (x, x_recon), (log_vq, log_disc), metrics
    
    def train_loop(self):
        logger = self.logger
        logger.info('[g][Train][/g] Start training ...')
        logger.info(f'[g][Train][/g] Training steps per epoch: {len(self.train_dataloader)}')
        logger.info(f'[g][Train][/g] Validation steps per epoch: {len(self.val_dataloader)}')
        logger.info(f'[g][Train][/g] Max training steps: {self.train_cfg.max_train_steps}')
        logger.info(f'[g][Train][/g] Save state every {self.train_cfg.save_state_every} steps')
        logger.info(f'[g][Train][/g] Validation every {self.train_cfg.val_every} steps')
        logger.info(f'[g][Train][/g] Gradient accumulation steps: {self.accelerator.gradient_accumulation_steps}')
        
        for x in self.get_train_input():
            x_recon = self.train_step(x)
            
            if self.train_n_iter >= self.train_cfg.max_train_steps:
                logger.info(f'[g][Train][/g] Training steps reached max steps, save state weights and exit ...')
                self.save_state()
                break
            
            # save ckpt
            if self.train_n_iter % self.train_cfg.save_state_every == 0:
                self.save_state()
                
            # training images
            if self.train_n_iter % self.train_cfg.save_train_every == 0:
                x_vis, x_recon_vis = self.to_zero_one(x), self.to_zero_one(x_recon)
                self.logger.log_images([x_vis, x_recon_vis], 4, ['train/gt', 'train/recon'], 'fusion', self.train_n_iter)
                
            # validation
            if self.train_n_iter % self.train_cfg.val_every == 0:
                # close training tbar temporarily
                if self.accelerator.is_main_process and not self.cfg.debug:
                    self.tbar.update(self.train_iter_task, visible=False)
                    self.tbar.reset(self.val_iter_task)
                
                # validation loop
                mean_metrics = self.val_loop()
                self.train_state.update(mode='val')
            
                # save ema
                save_or_not = self.save_checker(mean_metrics)
                if save_or_not:
                    self.save_ema()
    
    @torch.no_grad()
    def val_loop(self):
        logger = self.logger
        accelerator = self.accelerator
        tbar = self.tbar
        self.accelerator.wait_for_everyone()
        logger.info('[b][Validation][/b] Start validation ...')
        
        mean_metrics = {'psnr': 0., 'ssim': 0.}
        mean_vq_log = {}
        mean_disc_log = {}
        for n_iter, x in enumerate(self.val_dataloader, 1):
            x = x[0]
            x = self.to_neg_one_one(x)
            (x, x_recon), (log_vq, log_disc), metric = self.val_step(x)
            
            # metrics
            mean_metrics = self.mean_dicts(mean_metrics, metric, n_iter)
            mean_vq_log = self.mean_dicts(mean_vq_log, log_vq, n_iter)
            mean_disc_log = self.mean_dicts(mean_disc_log, log_disc, n_iter)
            
            # update progress bar
            if accelerator.is_main_process and not self.cfg.debug:
                # tbar.reset(self.val_iter_task)
                tbar.update(self.val_iter_task, total=len(self.val_dataloader), completed=n_iter, visible=True,
                            description=f'Validation Iter [{n_iter}/{len(self.val_dataloader)}] - PSNR: {metric["psnr"]}, SSIM: {metric["ssim"]}')
                
        # distributed mean
        mean_metrics = self.distributed_mean_dict(mean_metrics)
        mean_vq_log = self.distributed_mean_dict(mean_vq_log)
        mean_disc_log = self.distributed_mean_dict(mean_disc_log)
            
        # log mean metrics
        logger.log_curves(mean_metrics, self.val_n_iter)
        logger.log_curves(mean_vq_log, self.val_n_iter)
        logger.log_curves(mean_disc_log, self.val_n_iter)
        
        logger.info('[g]Validation Mean Metrics[/g] - PSNR: {:.4f}, SSIM: {:.4f}'.format(mean_metrics['psnr'], mean_metrics['ssim']))
        logger.info('================================================================')
        
        # log images
        x_vis = self.to_zero_one(x).detach().cpu()
        x_recon_vis = self.to_zero_one(x_recon).detach().cpu()
        self.logger.log_images([x_vis, x_recon_vis], 4, ['val/gt', 'val/recon'], 'fusion', self.val_n_iter)
        
        # clear
        del x_vis, x_recon_vis
        torch.cuda.empty_cache()
        gc.collect()
        
        return mean_metrics
    
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
            
    def mean_dicts(self, mean_d: dict, new_d: dict, n_iter: int):
        for k, v in new_d.items():
            if k not in mean_d:
                assert n_iter == 1, 'The first iter should not be averaged'
                mean_d[k] = v
            mean_d[k] = mean_d[k] * (n_iter - 1) / n_iter + v / n_iter
        
        return mean_d
    
    def get_last_layer(self, use_ema: bool=False):
        if use_ema:
            return self.accelerator.unwrap_model(self.ema_decoder).ema_model.conv_out.weight
        else:
            return self.accelerator.unwrap_model(self.decoder).conv_out.weight

## helper functions

def experimental_logger(exp_name: str='VQ_LFQ', config_args: NameSpace=None):
    state = PartialState()
    
    config_args.full_arch = exp_name  # hack for NoneLogger
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
        args.output_dir = args.save_model_path = [None]
    
    if accelerator.use_distributed:
        (args.output_dir, args.save_model_path) = accelerate.utils.broadcast_object_list([args.output_dir, args.save_model_path])
    accelerator.project_configuration.project_dir = args.output_dir
    logger.info(f'model weights will be save at {args.output_dir}')
    
    return args
        
def main(args: NameSpace):
    accelerator = accelerate.Accelerator(mixed_precision='bf16',
                                         dataloader_config=DataLoaderConfiguration(split_batches=False,
                                                                                   even_batches=False,
                                                                                   non_blocking=True,
                                                                                   dispatch_batches=None,),
                                         project_config=ProjectConfiguration(project_dir='log_file/',
                                                                             total_limit=args.ckpt_max_limit,
                                                                             automatic_checkpoint_naming=True,),
                                         gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=args.optimizer_cfg.gradient_accumulation_steps,
                                                                                                 adjust_scheduler=False,
                                                                                                 sync_with_dataloader=False,
                                                                                                 sync_each_batch=False,),
                                         step_scheduler_with_optimizer=False,
                                         kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
                                         )
    
    # get logger
    logger = experimental_logger(exp_name=args.exp_name, 
                                 config_args=args)
    
    # specify weights path
    args = specify_weights_path(args, logger, accelerator)
    
    # set cuda device
    torch.cuda.set_device(f'cuda:{accelerator.process_index}')
    
    # dataset
    (train_ds, train_dl, val_ds, val_dl), args = get_train_dataset(args)
    
    # init model
    logger.info(f'Init LFQ VQ VAE to experiment: {args.exp_name}')
    engine = LFQ_VQ_VAE_Trainer(cfg=args,
                                logger=logger, 
                                accelerator=accelerator,
                                train_dataloader=train_dl,
                                val_dataloader=val_dl)
    
    # train it
    engine.train_loop()
    
    # end of training
    accelerator.end_training()    
    

if __name__ == "__main__":
    from utils.log_utils import LoguruLogger
    import sys
    logger = LoguruLogger.logger(sink=sys.stdout)
    LoguruLogger.add('log_file/running_traceback.log', format="{time:MM-DD hh:mm:ss} {level} {message}", 
                    level="WARNING", backtrace=True, diagnose=True, mode='w')
    LoguruLogger.add(sys.stderr, format="{time:MM-DD hh:mm:ss} {level} {message}", 
                    level="ERROR", backtrace=True, diagnose=True)
    
    ## main
    args = parse_args()
    model_cfg = NameSpace.init_from_yaml(args.config, end_with='')
    args = NameSpace.merge_parser_args(args, model_cfg)
    
    state = PartialState()
    accelerate.utils.set_seed(2025)
    
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
    
    
