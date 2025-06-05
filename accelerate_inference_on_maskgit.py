import os
import argparse
from shutil import which
import torch as th
import torch.nn.functional as F
from einops import rearrange, repeat
import accelerate
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from torchvision.io import read_image, write_png, ImageReadMode

from model.lfq_vae.autoencoder import Decoder, Encoder
from model.lfq_vae.quantizer import LFQ_v0 as LFQ
from model.maskbit.maskbit_binary_model import LFQBert as MaskBit
from utils import NameSpace, easy_logger, catch_any_error

logger = easy_logger(func_name='maskgit_inference')

CACHE_DIR = '/Data3/cao/ZiHanCao/exps/panformer/model/text_process_pipe/pretrained_ckpts'
class HFEmbedder(th.nn.Module):
    def __init__(self, version: str, max_length: int, weight_ckpt_dir: str=CACHE_DIR, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length, cache_dir=weight_ckpt_dir)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length, cache_dir=weight_ckpt_dir)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> th.Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key].type(th.float32)

class MaskGiTInferencer:
    def __init__(self, cfg: NameSpace):
        self.cfg = cfg
        self.llm_cfg = cfg.llm_cfg
        self.vae_cfg = cfg.vae_cfg
        self.maskgit_cfg = cfg.maskgit_cfg
        self.infer_cfg = cfg.infer_cfg
        
        self.llm = HFEmbedder(self.llm_cfg.llm_model_name,
                              self.llm_cfg.max_length, 
                              self.llm_cfg.weight_ckpt_dir).cuda()
        self.vae_encoder = Encoder(**self.vae_cfg.encoder_cfg).cuda()
        self.vae_decoder = Decoder(**self.vae_cfg.decoder_cfg).cuda()
        self.lfq = LFQ(**self.vae_cfg.quantizer_cfg).cuda()
        self.maskgit = MaskBit(**self.maskgit_cfg.model_cfg).cuda()
        
    def load_ckpts(self):
        self.vae_encoder = self.vae_encoder.to(self.accelerator.device)
        self.vae_decoder = self.vae_decoder.to(self.accelerator.device)
        self.lfq = self.lfq.to(self.accelerator.device)
        self.maskgit = self.maskgit.to(self.accelerator.device)
        
        # load checkpoints
        if self.vae_cfg.pretrained_ckpt.endswith('.ckpt'):  # is official checkpoint
            ckpt = th.load(self.vae_cfg.pretrained_ckpt, weights_only=True)['state_dict']
            enc_params = {}
            dec_params = {}
            for k, v in ckpt.items():
                if k.startswith('encoder'):
                    enc_params[k.replace('encoder.', '')] = v
                elif k.startswith('decoder'):
                    dec_params[k.replace('decoder.', '')] = v
            self.vae_encoder.load_state_dict(enc_params)
            self.vae_decoder.load_state_dict(dec_params)
        else:  # is post-trained checkpoint
            ckpt = accelerate.utils.load_state_dict(self.vae_cfg.pretrained_ckpt)
            self.vae_encoder.load_state_dict(ckpt['ema_encoder']['ema_model'])
            self.vae_decoder.load_state_dict(ckpt['ema_decoder']['ema_model'])
            self.lfq.load_state_dict(ckpt['ema_lfq']['ema_model']) if 'ema_lfq' in ckpt else None
        print('load vae ckpt done')
        
        # load maskgit
        accelerate.load_checkpoint_in_model(
            self.maskgit,
            self.maskgit_cfg.model_ckpt_path,
            strict=True
        )
        print('load maskgit ckpt done')
        
    def encode_instruction(self, instruction: str | list[str]):
        if isinstance(instruction, str):
            instruction = [instruction]
        # TODO: text guidance may use float32
        return self.llm(instruction)
    
    def encode_quantized_image(self, image: th.Tensor):
        # image: [bs, c, h, w]
        # to (-1, 1)
        image = 2 * image - 1
        image_codes = self.vae_encoder(image)
        image_codes, _, _ = self.lfq(image_codes, return_loss_breakdown=False)
        return image_codes.float()
    
    def make_simple_vq_model_to_sample(self, token_size: list[int]):
        encoder = self.vae_encoder
        decoder = self.vae_decoder
        quantizer = self.quantizer
        split_m = self.split_m
        
        class SimpleVQSampleModel(th.nn.Module):
            def __init__(self, encoder, decoder, quantizer, token_size: list[int], split_m: int):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.quantizer = quantizer
                self.token_size = token_size
                self.split_m = split_m
                
            def forward(self, x: th.Tensor):
                return self.decode_tokens(x)
            
            def decode_tokens(self, decoded_codes: th.Tensor):
                # if just random mask tokens
                # decoded_codes = self.quantizer.indices_to_codes(predicted_indices)
                decoded_codes = rearrange(decoded_codes, 'b (h w) c -> b c h w',
                                            h=self.token_size[0], w=self.token_size[1]).float()
                decoded_img = self.decoder(decoded_codes)
                return decoded_img
        
        return SimpleVQSampleModel(encoder, decoder, quantizer, token_size, split_m)
    
    def decode_image(self, image_codes: th.Tensor):
        image_decoded = self.vae_decoder(image_codes)
        # to (0, 1)
        image_decoded = (image_decoded + 1) / 2
        image_decoded = th.clamp(image_decoded, 0, 1).type(th.float32)
        
        return image_decoded

    def img_type_shape_check(self, img: th.Tensor):
        if img.ndim == 3:  # RGB image with c-last
            img = img.unsqueeze(0)
        elif img.ndim == 2:  # gray image
            img = img.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 3)
        
        # channel check
        if img.shape[1] != 3:
            img = rearrange(img, 'b h w c -> b c h w')
        
        assert img.ndim == 4, 'image must be a 4D tensor'
        assert img.shape[1] == 3, 'image must have 3 channels'
        
        return img
    
    def load_image(self, img_path: str):
        # force to read as RGB
        img = read_image(img_path, ImageReadMode.RGB) / 255.
        return self.img_type_shape_check(img).float().cuda()
    
    def lfq_stride_check(self, image: th.Tensor, stride: int=16, resize: bool=False):
        if check_fail := image.shape[-1] % stride != 0 or image.shape[-2] % stride != 0:
            logger.warning(f'input image size {image.shape[-1]}x{image.shape[-2]} is not divisible by {stride}, \
                which may cause output size mismatch')
            if resize:
                orig_size = (image.shape[-1], image.shape[-2])
                # find the nearest size that is divisible by stride (larger)
                image = F.interpolate(image, 
                                      size=(((image.shape[-2] + stride - 1) // stride) * stride,
                                            ((image.shape[-1] + stride - 1) // stride) * stride),
                                      mode='nearest')
                logger.warning(f'resize: {orig_size} -> {tuple(image.shape[-2:])}')
        
        def inverse_lfq_stride_check(image: th.Tensor):
            if check_fail and resize:
                image = F.interpolate(image, 
                                      size=orig_size,
                                      mode='nearest')
                # logger.warning(f'inverse resize: {tuple(image.shape[-2:])} -> {orig_size}')
            return image

        return image, inverse_lfq_stride_check
    
    def inference_by_input_path(self, 
                                degraded_s1: str, 
                                degraded_s2: str, 
                                instruction: str | list[str]):
        # load image
        degraded_s1 = self.load_image(degraded_s1)
        degraded_s2 = self.load_image(degraded_s2)
        assert degraded_s1.shape == degraded_s2.shape, 'two degraded images must have the same shape'
        
        # lfq stride check
        degraded_s1, _ = self.lfq_stride_check(degraded_s1)
        degraded_s2, inverse_lfq_stride_check = self.lfq_stride_check(degraded_s2)
        
        # encode image
        degraded_s1_codes = self.encode_quantized_image(degraded_s1)
        degraded_s2_codes = self.encode_quantized_image(degraded_s2)
        
        # encode instruction
        instruction = self.encode_instruction(instruction)
        
        # inference
        fused_clean_img, _ = self.maskgit.sample(
            vq_model=self.make_simple_vq_model_to_sample(self.maskgit_cfg.sample_size),
            conditions=((degraded_s1_codes, degraded_s2_codes), instruction),
            num_samples=1,
            num_steps=self.maskgit_cfg.num_steps,
            img_size=self.maskgit_cfg.sample_size,
            codebook_size=self.lfq.codebook_size,
            codebook_splits=self.maskgit_cfg.split_m,
            gumbel_temperature=self.maskgit_cfg.gumbel_temperature
        )
        
        return inverse_lfq_stride_check(fused_clean_img)
    
    def inference_by_input_tensor(self, 
                                  degraded_s1_codes: th.Tensor,
                                  degraded_s2_codes: th.Tensor,
                                  encoded_instruction: th.Tensor):
        # inference
        fused_clean_img, _ = self.maskgit.sample(
            vq_model=self.make_simple_vq_model_to_sample(self.maskgit_cfg.sample_size),
            conditions=((degraded_s1_codes, degraded_s2_codes), encoded_instruction),
            num_samples=1,
            num_steps=self.maskgit_cfg.num_steps,
            img_size=self.maskgit_cfg.sample_size,
            codebook_size=self.lfq.codebook_size,
            codebook_splits=self.maskgit_cfg.split_m,
            gumbel_temperature=self.maskgit_cfg.gumbel_temperature
        )
        
        return fused_clean_img
    
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--wds_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--image_degraded_path1', type=str)
    parser.add_argument('--image_degraded_path2', type=str)
    parser.add_argument('--instruction', type=str)
    return parser.parse_args()

def inference_from_wds(cfg: NameSpace, inferencer: MaskGiTInferencer):
    from task_datasets.DIF.degraded_wds import get_degraded_wds_loader
    
    wds_path = cfg.wds_path
    output_dir = cfg.infer_cfg.output_dir
    wds_dl_kwargs = cfg.infer_cfg.wds_dl_kwargs
    
    # make output dir
    os.makedirs(output_dir, exist_ok=True)
    
    _, dataloader = get_degraded_wds_loader(wds_path, **wds_dl_kwargs)
    
    for batch in dataloader:
        instruction_text = batch['info.json']
        file_name = batch['__key__']
        print(f'inference with instruction: {instruction_text} on {file_name} ...')
        degraded_s1_codes = batch['encoded_m1_degraded']
        degraded_s2_codes = batch['encoded_m2_degraded']
        encoded_instruction = batch['encoded_instruction']
        fused_clean_img = inferencer.inference_by_input_tensor(degraded_s1_codes, degraded_s2_codes, encoded_instruction)
        
        # save
        for fuse_img in fused_clean_img:
            fuse_img = fuse_img.clamp(0, 1) * 255.
            write_png(fuse_img.type(th.uint8), os.path.join(output_dir, f'{file_name}.png'))
        
def inference_from_image_path(cfg: NameSpace, inferencer: MaskGiTInferencer):
    image_degraded_path1 = cfg.infer_cfg.image_degraded_path1
    image_degraded_path2 = cfg.infer_cfg.image_degraded_path2
    instruction = cfg.infer_cfg.instruction
    output_dir = cfg.infer_cfg.output_dir
    
    # make output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # inference
    file_name = os.path.basename(image_degraded_path1)
    fused_clean_img = inferencer.inference_by_input_path(image_degraded_path1, image_degraded_path2, instruction)

    # save
    fuse_img = fused_clean_img[0].clamp(0, 1) * 255.
    write_png(fuse_img.type(th.uint8), os.path.join(output_dir, f'{file_name}.png'))
    
    
if __name__ == '__main__':
    args = get_args()
    cfg = NameSpace.init_from_yaml(args.config)
    
    #* inference
    with catch_any_error():
        inferencer = MaskGiTInferencer(cfg)
        
        if args.wds_path is not None:
            inference_from_wds(cfg, inferencer)
        else:
            assert (args.image_degraded_path1 is not None 
                    and args.image_degraded_path2 is not None 
                    and args.instruction is not None), \
                'image_degraded_path1, image_degraded_path2, and instruction must be provided'
            inference_from_image_path(cfg, inferencer)
        