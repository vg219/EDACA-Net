"""
Real world degradation pipeline for image fusion tasks

Pipelines includes:
    - high exposure
    - low exposure
    - high contrast
    - low contrast
    - gaussian blur
    - motion blur
    - gaussian noise
    - rain
    - snow
    - haze
    - jpeg
    - downsample

Provide some instructions for each degradation pipeline,
provide LLM encoding and VQ VAE encoding for each pipeline,
and save them all in webdataset files.

LLM includes:
    - T5
    - CLIP
    
VQ VAE includes:
    - LFQ
    
--------------------------------------------------------------------------------

Authors: Zihan Cao, Yu Zhong
Date: 2024-12-06
Email: iamzihan666@gmail.com

GLP v3 license
Copyright (C) 2024 Zihan Cao, Yu Zhong, Mathematical School, University of Electronic Science and Technology of China (UESTC)

"""


import os
import numpy as np
import torch as th
from PIL import Image
from collections import namedtuple
from copy import deepcopy
import warnings
import json
from pathlib import Path
import webdataset as wds
from tqdm import tqdm
import io
from typing import Callable
import kornia.augmentation as Ka
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.random_generator import PlainUniformGenerator
from kornia.augmentation.container.params import ParamItem
from kornia.geometry.transform import resize
from kornia.io import load_image, write_image, ImageLoadType
from albumentations import RandomRain, RandomFog
from torchmetrics.image import PeakSignalNoiseRatio
import warnings
os.environ['CURL_CA_BUNDLE'] = ''  # hack for transformers ssl error

from task_datasets.DIF.rainy_img_create import synthetic_rainy_pair
from task_datasets.DIF.haze_img_create import hazy_simu_kornia as synthetic_haze_pair


#* some prefix constants and flags ==================================================
# rain drop dir
# FIXME: find one way to apply horizontal or vertical rain drops
RAIN_DROP_DIR = '/Data3/cao/ZiHanCao/datasets/RainMask/horizontal'
VQ_STRIDE = 16

#* pipeline parameters to tune ==================================================
IS_ON_VIS_IR_MODALITIES: bool = True

#! haze depth map path, we use as global variable, since kornia does not support input
#! two inputs that does not exist in the DATA_KEYS
# FIXME: this way is not good, find a better way to handle this
HAZE_IN_PIPE_FLAG = False

# it means ir image input into the pipeline, the rational is that
# if haze in vis image, it cause foggy weather, but in ir image, it is not
# it often cause the ir image to be blur
# used in class `RandomHazeRealWorld`
INFRAFED_IMG_IN_PIPE_FLAG: bool = False
HAZE_DEPTH_MAP_PATH: str | None = None

#* some instructions =======================================================
instructions = json.load(open('task_datasets/prompts/restoration_prompts.json', 'r'))

random_ambiguous = float(os.getenv('RANDOM_AMBIGUOUS', '0.1'))
print(f'set `random_ambiguous` to {random_ambiguous}')

# pipeline return
PipeReturn = namedtuple('PipeReturn', ['degradation', 'instruction', 'task', 'ambiguous'])
paired_data = namedtuple('paired_data', ['instruction', 'vis', 'inf', 'vis_gt', 'inf_gt'])

# warning class definition
class InputNotDivisibleByVQStrideWarning(RuntimeWarning):
    pass

def random_instruction(*task: str, random_ambiguous: float=random_ambiguous):
    if random_ambiguous > np.random.rand():
        return {'instruction': np.random.choice(instructions['ambiguous']), 'task': task[0], 'ambiguous': True}
    else:
        _inst = deepcopy(instructions)
        for t in task:
            if t in _inst:
                _inst = _inst[t]
            else:
                raise ValueError(f'no instruction for {t}')
        return {'instruction': np.random.choice(_inst), 'task': task[0], 'ambiguous': False}
    
#* utilities =============================================================

def kornia_tensor_to_numpy(tensor: th.Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        assert tensor.shape[0] == 1, 'batch size must be 1'
        tensor = tensor.squeeze(0)
    return tensor.permute(1, 2, 0).numpy()

def numpy_to_kornia_tensor(image: np.ndarray) -> th.Tensor:
    return th.from_numpy(image).permute(2, 0, 1)[None]

def normalize_img(img: th.Tensor):
    img_max = img.max()
    img_min = img.min()
    return (img - img_min) / (img_max - img_min)

def tensor_img_to_bytes(img: th.tensor) -> bytes:
    """
    convert a tensor to bytes.
    """
    if img.max() > 1 or img.min() < 0:
        img = normalize_img(img)
    
    img = img.squeeze()
    img = (Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))).convert('RGB')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    
    return img_bytes.getvalue()

def numpy_img_to_bytes(img: np.ndarray) -> bytes:
    if img.max() > 1 or img.min() < 0:
        img = normalize_img(img)
    
    img = (Image.fromarray((img * 255).astype(np.uint8))).convert('RGB')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    
    return img_bytes.getvalue()

def set_global_haze_depth_map_path(pipe_img_path: str | Path):
    global HAZE_DEPTH_MAP_PATH, HAZE_IN_PIPE_FLAG
    if not HAZE_IN_PIPE_FLAG:
        return
    
    if isinstance(pipe_img_path, str):
        pipe_img_path = Path(pipe_img_path)
    pipe_img_path: Path
    HAZE_DEPTH_MAP_PATH = pipe_img_path.parents[1] / 'depth' / (pipe_img_path.stem + '_depth.png')
    assert HAZE_DEPTH_MAP_PATH.exists(), f'depth map not found at {HAZE_DEPTH_MAP_PATH}'
    HAZE_DEPTH_MAP_PATH = HAZE_DEPTH_MAP_PATH.as_posix()

#* some kornia random generator ================================================

class RandomPathsGenerator(RandomGeneratorBase):
    def __init__(self, rain_drop_dir: str | None = None):
        super().__init__()
        self.rain_drop_dir = rain_drop_dir
        
    def make_samplers(self, device: th.device, dtype: th.dtype):
        self.rain_drop_paths = list(Path(self.rain_drop_dir).glob('*'))
        assert len(self.rain_drop_paths) > 0, f'no rain drop images found in {self.rain_drop_dir}'
        print(f'found {len(self.rain_drop_paths)} rain drop images in {self.rain_drop_dir}')
        
    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False):
        batch_size = batch_shape[0]
        
        if self.rain_drop_paths is None:
            return {'rain_drop_path': None}
        
        if same_on_batch:
            rain_drop_path = [np.random.choice(self.rain_drop_paths)] * batch_size
        else:
            rain_drop_path = [np.random.choice(self.rain_drop_paths) for _ in range(batch_size)]
            
        return {'rain_drop_path': rain_drop_path}

#* some kornia augmentation ==================================================

class RandomDownsample(GeometricAugmentationBase2D):
    def __init__(self, down_scale: list[float]=[1, 2.], p: float=1.):
        super().__init__(p=p, p_batch=1., same_on_batch=False, keepdim=False)
        self.down_scale = down_scale
        
    def apply_transform(self, input, params, flags, transform=None):
        scale = np.random.uniform(self.down_scale[0], self.down_scale[1])
        shape = (int(input.shape[2] // scale), int(input.shape[3] // scale))
        img_down = resize(
            input,
            shape,
            interpolation='bilinear',
            side='short',
            align_corners=True,
        )
        img_up = resize(
            img_down,
            (input.shape[2], input.shape[3]),
            interpolation='bilinear',
            side='short',
            align_corners=True,
        )
        return img_up
    
    def compute_transformation(self, input, params, flags):
        return None
    
class RandomRainAdpotAlbumentations(IntensityAugmentationBase2D):
    def __init__(self,             
                 slant_range=(-20, 20),
                 drop_length=20,
                 drop_width=3,
                 drop_color=(225, 225, 225),
                 blur_value=1.,
                 brightness_coefficient=1.,
                 rain_type=None,
                 always_apply: bool | None = None,
                 p: float = 1.):
        super().__init__(p=p, p_batch=1., same_on_batch=False, keepdim=False)
        self.rain_op = RandomRain(
            slant_range=slant_range,
            drop_length=drop_length,
            drop_width=drop_width,
            drop_color=drop_color,
            blur_value=blur_value,
            brightness_coefficient=brightness_coefficient,
            rain_type=rain_type,
            always_apply=always_apply,
            p=p,
        )
        
    def apply_transform(self, input, params, flags, transform=None):
        if th.is_tensor(input):
            if input.ndim == 4:
                assert input.shape[0] == 1, 'batch size must be 1'
                input = input.squeeze(0)
            input = input.permute(1, 2, 0)
            input = input.numpy()
        rainy = self.rain_op(image=input)['image']
        return th.from_numpy(rainy).permute(2, 0, 1)[None]
    
class RandomFogAdpotAlbumentations(IntensityAugmentationBase2D):
    def __init__(self,             
                 alpha_coef: float = 0.08,
                 fog_coef_range: tuple[float, float] = (0.3, 1),
                 always_apply: bool | None = None,
                 p: float = 1.):
        super().__init__(p=p, p_batch=1., same_on_batch=False, keepdim=False)
        self.fog_op = RandomFog(
            alpha_coef=alpha_coef,
            fog_coef_range=fog_coef_range,
            always_apply=always_apply,
            p=p,
        )

    def apply_transform(self, input, params, flags, transform=None):
        if th.is_tensor(input):
            if input.ndim == 4:
                assert input.shape[0] == 1, 'batch size must be 1'
                input = input.squeeze(0)
            input = input.permute(1, 2, 0)
            input = input.numpy()
        rainy = self.fog_op(image=input)['image']
        return th.from_numpy(rainy).permute(2, 0, 1)[None]

class RandomRainRealWorld(IntensityAugmentationBase2D):
    def __init__(self, p: float=1., rain_drops_dir: str | None = RAIN_DROP_DIR):
        super().__init__(p=p, p_batch=1., same_on_batch=False, keepdim=False)
        self._param_generator = RandomPathsGenerator(rain_drops_dir)
        
    def apply_transform(self, input, params, flags, transform=None):
        input = kornia_tensor_to_numpy(input)
        rainy = synthetic_rainy_pair(input, params['rain_drop_path'])
        return numpy_to_kornia_tensor(rainy)

class RandomHazeRealWorld(IntensityAugmentationBase2D):
    def __init__(self, p: float=1., haze_distance: tuple[float, float]=(3, 15), is_vis_ir_modalities: bool=False):
        super().__init__(p=p, p_batch=1., same_on_batch=False, keepdim=False)
        self.haze_distance = haze_distance
        self._param_generator = PlainUniformGenerator((haze_distance, 'haze_distance', None, None))
        self.is_vis_ir_modalities = is_vis_ir_modalities
        if is_vis_ir_modalities:
            self.ir_blur_module = Ka.RandomGaussianBlur(kernel_size=(5, 17), sigma=(1.0, 2.0), p=1.)
        
    def apply_transform(self, input, params, flags, transform=None):
        global HAZE_DEPTH_MAP_PATH, INFRAFED_IMG_IN_PIPE_FLAG
        
        if not INFRAFED_IMG_IN_PIPE_FLAG:
            # only support one image input
            assert input.ndim == 4 and input.shape[0] == 1, 'only support one image input'
            input = input.squeeze(0)
            # read depth map, take from the global variable, handle the pixel value range (depth) in function 
            assert HAZE_DEPTH_MAP_PATH is not None, 'haze depth map path is not set'
            depths = load_image(HAZE_DEPTH_MAP_PATH, ImageLoadType.UNCHANGED).squeeze(0).float()
            # airlight: night: 0.13, day: 0.34
            from kornia.color import rgb_to_grayscale
            airlight = rgb_to_grayscale(input).mean() * 4
            hazy = synthetic_haze_pair(input, depths, visual_range=params['haze_distance'], airlight=airlight.item())
            hazy = hazy.unsqueeze(0)
            return hazy
        elif self.is_vis_ir_modalities and INFRAFED_IMG_IN_PIPE_FLAG:
            # FIXME: random gaussian blur here, may fix kernel size or sigma by the intensity of haze
            
            blur_ir = self.ir_blur_module(input)
            return blur_ir
        else:
            raise ValueError(f'haze real world augmentation is not supported when {INFRAFED_IMG_IN_PIPE_FLAG=} and {self.is_vis_ir_modalities=}')

##* degradation pipes =====================================================

high_exposure_pipe = lambda: Ka.RandomBrightness(brightness=(1.1, 1.5), p=1.)
low_exposure_pipe = lambda: Ka.RandomBrightness(brightness=(0.5, 0.9), p=1.)
gaussian_blur_pipe = lambda: Ka.RandomGaussianBlur(kernel_size=(5, 31), sigma=(1.0, 3.0), p=1.)
motion_blur_pipe = lambda: Ka.RandomMotionBlur(kernel_size=(5, 31), angle=35., direction=0.5, p=1.)
gaussian_noise_pipe = lambda: Ka.RandomGaussianNoise(mean=0., std=15 / 255, p=1.)
rain_pipe = lambda: Ka.RandomRain(drop_height=(5, 30), drop_width=(1, 5), p=1.)
rain_album_pipe = lambda: RandomRainAdpotAlbumentations(p=1., drop_color=(235, 235, 235), blur_value=0.03)
rain_real_world_pipe = lambda: RandomRainRealWorld(p=1.)
haze_album_pipe = lambda: RandomFogAdpotAlbumentations(p=1.)
haze_real_world_pipe = lambda: RandomHazeRealWorld(p=1., is_vis_ir_modalities=IS_ON_VIS_IR_MODALITIES)
snow_pipe = lambda: Ka.RandomSnow(snow_coefficient=(0.15, 0.25), brightness=(3.0, 5.0), p=1.)
jpeg_pipe = lambda: Ka.RandomJPEG(jpeg_quality=(5, 50), p=1.)
low_contrast_pipe = lambda: Ka.RandomContrast(contrast=(0.2, 0.8), p=1.)
high_contrast_pipe = lambda: Ka.RandomContrast(contrast=(1.6, 2.6), p=1.)
downsample_pipe = lambda: RandomDownsample(down_scale=[1, 2.], p=1.)

#* extract param info ========================================================

def extract_param_info(param: ParamItem):
    info = deepcopy(param.data)
    info.pop('batch_prob', None)
    info.pop('forward_input_shape', None)
    return info

def make_degradation_string(param_data):
    return ', '.join([f'{k}: {v.item() if th.is_tensor(v) and v.numel() == 1 else v}' for k, v in param_data.items()])
    
pipe_instruction_dict = {
    'RandomBrightness': lambda param: PipeReturn(
        degradation=f'brightness: {make_degradation_string(extract_param_info(param))}',
        **(random_instruction('lighting_adjustment', 'brightness', 'brightening') \
            if param.data['brightness_factor'] < 1 \
            else random_instruction('lighting_adjustment', 'brightness', 'darkening')
        )
    ),
    'RandomGaussianBlur': lambda param: PipeReturn(
        degradation=f'gaussian blur: {make_degradation_string(extract_param_info(param))}',
        **random_instruction('blur_removal', 'gaussian_blur')
    ),
    'RandomMotionBlur': lambda param: PipeReturn(
        degradation=f'motion blur: {make_degradation_string(extract_param_info(param))}',
        **random_instruction('blur_removal', 'motion_blur')
    ),
    'RandomGaussianNoise': lambda param: PipeReturn(
        degradation=f'gaussian noise: {make_degradation_string(extract_param_info(param))}',
        **random_instruction('noise_effects', 'gaussian_noise')
    ),
    'RandomRain': lambda param: PipeReturn(
        degradation=f'rain: {make_degradation_string(extract_param_info(param))}',
        **random_instruction('weather_effects', 'rain')
    ),
    'RandomRainRealWorld': lambda param: PipeReturn(
        degradation=f'rain: real world',
        **random_instruction('weather_effects', 'rain')
    ),
    'RandomHazeRealWorld': lambda param: PipeReturn(
        degradation=f'haze: {make_degradation_string(extract_param_info(param))}',
        **random_instruction('weather_effects', 'haze')
    ),
    'RandomSnow': lambda param: PipeReturn(
        degradation=f'snow: {make_degradation_string(extract_param_info(param))}',
        **random_instruction('weather_effects', 'snow')
    ),
    'RandomJPEG': lambda param: PipeReturn(
        degradation=f'jpeg: {make_degradation_string(extract_param_info(param))}',
        **random_instruction('image_quality', 'jpeg_compression')
    ),
    'RandomContrast': lambda param: PipeReturn(
        degradation=f'contrast: {make_degradation_string(extract_param_info(param))}',
        **(random_instruction('contrast', 'low_contrast') \
            if param.data['contrast_factor'] < 1 \
            else random_instruction('contrast', 'high_contrast')
        )
    ),
    'RandomDownsample': lambda param: PipeReturn(
        degradation=f'downsample: {make_degradation_string(extract_param_info(param))}',
        **random_instruction('blur_removal', 'super_resolution')
    )
}

def make_degradation_pipe(random_apply: int=2, n_inputs: int=1) -> Ka.AugmentationSequential:
    assert 0 < random_apply <= 7, 'random_apply must be in [1, 7]'
    pipe = Ka.AugmentationSequential(
        # gaussian_blur_pipe(),
        # motion_blur_pipe(),
        
        # downsample_pipe(),
        
        # gaussian_noise_pipe(),
        # low_exposure_pipe(),
        # high_exposure_pipe(),
        
        # rain_pipe(),
        
        # rain_album_pipe(),
        # fog_album_pipe(),
        # snow_pipe(),
        
        # rain_real_world_pipe(),
        haze_real_world_pipe(),
        
        # jpeg_pipe(),
        # low_contrast_pipe(),
        # high_contrast_pipe(),
        
        data_keys=['input'] * n_inputs,
        random_apply=random_apply,
        same_on_batch=False,
    )
    
    # set some flags
    for p in pipe:
        if isinstance(p, RandomHazeRealWorld):
            global HAZE_IN_PIPE_FLAG
            HAZE_IN_PIPE_FLAG = True
    if HAZE_IN_PIPE_FLAG:
        print('[Pipe]: haze_real_world in pipe')
    
    return pipe

# image read and save

def read_image(path: str):
    img = load_image(path, ImageLoadType.RGB8)
    return img.unsqueeze(0) / 255.

def image_save_check(img: th.Tensor, require_data: str='numpy'):
    if img.ndim == 4:
        assert img.shape[0] == 1, 'batch size must be 1'
        img = img.squeeze(0)
    else:
        assert img.ndim == 3, f'image must be a 3D tensor, but got {img.ndim}'
        
    if img.dtype == th.float32 and img.max() <= 1.0:
        img = (img * 255).to(th.uint8)
    else:
        assert img.dtype == th.uint8, 'image must be a uint8 tensor'
    
    if require_data == 'numpy':
        # c-last
        return img.numpy().transpose(1, 2, 0)
    else:
        # c-first
        return img
    
def save_image(path: str | list[str], img: th.Tensor, backend: str='kornia'):
    require_data = 'numpy' if backend == 'PIL' else 'torch'
    img = image_save_check(img, require_data)
    if backend == 'kornia':
        def img_writer(p: str, i: th.Tensor):
            assert p.endswith('.jpg') or p.endswith('.jpeg'), 'only support jpg/jpeg format'
            write_image(p, i)
            print(f'save image to {p}')
    elif backend == 'PIL':
        def img_writer(p: str, i: th.Tensor):
            i = (i.squeeze(0).permute(1, 2, 0) * 255).to(th.uint8)
            Image.fromarray(i.numpy()).save(p, quality=100)
            print(f'save image to {p}')
    else:
        raise ValueError(f'backend {backend} not supported')
    
    if isinstance(path, str):
        assert img.ndim == 3, f'image must be a 3D tensor, but got {img.ndim}'
        img_writer(path, img)
    else:
        for p, i in zip(path, img):
            write_image(p, i)
            
# get pipe params and compose instruction

def get_pipe_params_and_instructions(pipe: Ka.AugmentationSequential):
    params_operated = pipe._params
    
    degrations = []
    instructions = []
    ambiguous = []
    
    assert len(params_operated) > 0, f'no params operated, params_operated: {params_operated}, pipe: {pipe}'
    
    for param in params_operated:
        pipe_name = param.name.split('_')[0]
        if pipe_name in pipe_instruction_dict:
            pipe_return = pipe_instruction_dict[pipe_name](param)
            degrations.append(pipe_return.degradation)
            instructions.append(pipe_return.instruction)
            ambiguous.append(pipe_return.ambiguous)
        else:
            warnings.warn(f'no instruction for {pipe_name} has params {param}')
                
    # compose the instruction
    if len(instructions) == 0 or len(params_operated) == 0:
        warnings.warn(f'no instruction found, pipe is {pipe}, '
                      f'has params: {params_operated}')
        return None, None
    else:
        if any(ambiguous):
            print('found ambiguous instruction')
            for i, am in enumerate(ambiguous):
                if am:
                    return degrations, instructions[i]
        
        instruction_composed = ''
        for i, ins in enumerate(instructions):
            if i == 0:
                instruction_composed += ins if len(instruction_composed) == 0 else f'{ins}, '
            elif i == len(instructions) - 1:
                instruction_composed += f' and {ins.lower()}'
            else:
                instruction_composed += f'{ins.lower()}, '
                                
    return degrations, instruction_composed


# pipeline apply policy

#TODO: add one modality is used pipe and the other is not used pipe (randomize the pipe usage)
def pipe_vis_ir_apply_policy(pipe: Ka.AugmentationSequential):
    """
    handle VIF degradation pipeline, the pipeline is different when infrared image is involved
    
    """
    def policy_pipe(img1: th.Tensor, img2: th.Tensor):
        # assume img1 is visible, img2 is infrared
        img1 = pipe(img1)
        _params = []
        params_1 = pipe._params
        for i, param in enumerate(params_1):
            pipe_name = param.name.split('_')[0]
            # assume rain and snow does not exist in infrared modality
            if pipe_name not in ['RandomRain', 'RandomRainRealWorld', 'RandomSnow']:
                _params.append(param)
                
            if pipe_name == 'RandomHazeRealWorld':
                global INFRAFED_IMG_IN_PIPE_FLAG
                INFRAFED_IMG_IN_PIPE_FLAG = True
                _params.append(param)
                
        img2 = pipe(img2, params=_params)
        pipe._params = params_1  # used to retrieve the degradation and instruction
        return img1, img2
    return policy_pipe

#* VQ or VAE encode function =======================================================

def make_encode_fn(encode_type: str='lfq_vae', half_precision: bool=False):
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()
    
    if encode_type == 'lfq_vae':
        from model.lfq_vae.autoencoder import Encoder, Decoder
        from model.lfq_vae.quantizer import LFQ_v0
        
        print('loading lfq vae encoder and decoder')
        
        # load encoder and decoder
        # f16 cfg
        encoder = Encoder(ch=128, z_channels=18, in_channels=3, ch_mult=(1, 1, 2, 2, 4), resolution=128, num_res_blocks=4).cuda()
        decoder = Decoder(ch=128, out_ch=3, z_channels=18, in_channels=3, ch_mult=(1, 1, 2, 2, 4), resolution=128, num_res_blocks=4).cuda()
        lfq = LFQ_v0(dim=18, codebook_size=2**18).cuda()
        
        # ckpt loading
        weight_d = th.load('/Data3/cao/ZiHanCao/exps/panformer/model/lfq_vae/ckpts/imagenet_256_L.ckpt', weights_only=True)['state_dict']

        enc_d = {}
        dec_d = {}
        for k, v in weight_d.items():
            if k.startswith('encoder'):
                enc_d[k.replace('encoder.', '')] = v
            elif k.startswith('decoder'):
                dec_d[k.replace('decoder.', '')] = v
        # print(enc_d.keys())
        # print(dec_d.keys())
        
        encoder.load_state_dict(enc_d)
        decoder.load_state_dict(dec_d)
        
        print('lfq encoder and decoder loaded')
                    
    else:
        raise ValueError(f'encode type {encode_type} not supported')
    
    encoder.eval()
    decoder.eval()
    lfq.eval()
    
    if half_precision:
        encoder = encoder.half()
        decoder = decoder.half()
        lfq = lfq.half()
    
    @th.no_grad()
    def lfq_encode_fn(img: th.Tensor):
        if img.ndim == 3:
            img = img.unsqueeze(0)
        assert img.ndim == 4 and img.shape[0] == 1, 'image must be a 4D tensor with batch size 1'
        assert img.shape[1] == 3, 'image must have 3 channels'
        
        if half_precision:
            img_in = img.half()  # half precision
        else:
            img_in = img
        img_in = img_in.cuda()  # to cuda
        img_for_psnr = img_in.clone()
        # to (-1, 1)
        img_in = 2 * img_in - 1
        
        # TODO: handle the not 16-divisible case
        if img_in.shape[-2] % VQ_STRIDE != 0 or img_in.shape[-1] % VQ_STRIDE != 0:
            warnings.filterwarnings('once', category=InputNotDivisibleByVQStrideWarning)
            warnings.warn("Input image dimensions are not divisible by 16, which may cause issues", 
                          category=InputNotDivisibleByVQStrideWarning)

        # encode
        z = encoder(img_in)
        z, indices, entropy_aux_loss = lfq(z, return_loss_breakdown=False)
        img_out = decoder(z)
        
        # to (0, 1)
        img_out = (img_out + 1) / 2
        img_out = th.clamp(img_out, 0, 1).type(th.float32)
        
        # psnr
        psnr = psnr_fn(img_out, img_for_psnr).item()
        
        # quantized to numpy
        z = z.squeeze(0).cpu().numpy().astype(np.int32)
        
        return z, psnr
    
    return lfq_encode_fn

#* LLM encode function ======================================================

import os
from einops import rearrange, repeat
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                        T5Tokenizer)
CACHE_DIR = '/Data3/cao/ZiHanCao/exps/panformer/model/text_process_pipe/pretrained_ckpts' #os.path.join(os.path.dirname(__file__), "pretrained_ckpts")
class HFEmbedder(th.nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length, cache_dir=CACHE_DIR)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length, cache_dir=CACHE_DIR)
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
    

def make_llm_encode_fn(llm_model: str='google-t5/t5-small',
                       max_length: int=77):
    if 't5' in llm_model:
        t5_encoder = HFEmbedder(llm_model, max_length=max_length, torch_dtype=th.bfloat16).cuda()
        print('t5 encoder loaded')
        
        def encode_instruction_fn(instruction: str | list[str]):
            if isinstance(instruction, str):
                instruction = [instruction]
            # TODO: text guidance may use float32
            return t5_encoder(instruction).cpu().numpy()
        
    elif 'clip' in llm_model:
        # "openai/clip-vit-large-patch14"
        clip_encoder = HFEmbedder(llm_model, max_length=77, torch_dtype=th.bfloat16).cuda()
        print('clip encoder loaded')
        
        def encode_instruction_fn(instruction: str | list[str]):
            if isinstance(instruction, str):
                instruction = [instruction]
            return clip_encoder(instruction)
        
    elif 'instructIR_llm' in llm_model:
        # TODO: re-use InstructIR light-weight LLM to encode instruction
        raise NotImplementedError('re-use InstructIR light-weight LLM to encode instruction, not implemented')
    else:
        raise ValueError(f'llm model {llm_model} not supported')
    
    return encode_instruction_fn
    
#* generate degradation instruction pairs ====================================

def generate_degradation_instruction_pairs_from_dir(
    pipe: Ka.AugmentationSequential,
    dir_path_p1: str, 
    dir_path_p2: str,
    gt_path: str,
    dataset_pattern: str,
    num_samples: int,
    pipe_apply_policy: "Callable | None" = None,
    vae_encode_fn: "Callable | None" = None,
    llm_encode_fn: "Callable | None" = None,
):
    dir_path_p1 = Path(dir_path_p1)
    dir_path_p2 = Path(dir_path_p2)
    dir_path_gt = Path(gt_path)
    
    assert dir_path_p1.exists() and dir_path_p1.is_dir(), f'dir {dir_path_p1} not exists'
    assert dir_path_p2.exists() and dir_path_p2.is_dir(), f'dir {dir_path_p2} not exists'
    assert dir_path_gt.exists() and dir_path_gt.is_dir(), f'dir {dir_path_gt} not exists'
    
    modality1_img_paths = list(dir_path_p1.glob('*'))
    modality2_img_paths = list(dir_path_p2.glob('*'))
    gt_img_paths = list(dir_path_gt.glob('*'))
        
    print(f'found {len(modality1_img_paths)} images in {dir_path_p1}')
    print(f'found {len(modality2_img_paths)} images in {dir_path_p2}')
    print(f'found {len(gt_img_paths)} images in {dir_path_gt}')
    
    assert len(modality1_img_paths) == len(modality2_img_paths) == len(gt_img_paths), \
        'two modalities and gt must have the same number of images'
    
    # sort the image paths
    modality1_img_paths.sort(key=lambda x: x.stem)
    modality2_img_paths.sort(key=lambda x: x.stem)
    gt_img_paths.sort(key=lambda x: x.stem)

    # wrap pipe apply policy
    if pipe_apply_policy is None:
        wrapped_pipe = pipe_vis_ir_apply_policy(pipe)
    else:
        wrapped_pipe = pipe
        
    # create saved dir
    saved_dir = Path(dataset_pattern).parent
    if not saved_dir.exists():
        saved_dir.mkdir(parents=True, exist_ok=True)
    print(f'saving dataset to {saved_dir}')
        
    # construct infinited image loaders
    import itertools
    def infinite_image_loader(*img_paths):
        return itertools.cycle(zip(*img_paths))
    
    # shard file restrict to 1GB
    with wds.ShardWriter(dataset_pattern, maxsize=1*1024*1024*1024) as sink:
        for n_sample_saved, (img_path_p1, img_path_p2, img_path_gt) in tqdm(enumerate(
            infinite_image_loader(modality1_img_paths, modality2_img_paths, gt_img_paths)
        ), total=num_samples, desc=f'generating degraded image pairs for {num_samples} ...'): 
            print(f'working on {img_path_p1.stem}')

            # read two modalities
            m1 = read_image(img_path_p1)
            m2 = read_image(img_path_p2)
            gt = read_image(img_path_gt)
            
            # apply pipe to degraded modalities
            degraded_m1, degraded_m2 = wrapped_pipe(m1, m2)
            
            # get pipe params and instructions
            degrations, instruction = get_pipe_params_and_instructions(pipe)
            info = {
                'degrations': degrations,
                'instruction': instruction,
                'H': m1.shape[-1],
                'W': m1.shape[-2],
            }

            # encode
            if vae_encode_fn is not None:
                encoded_m1, recon_psnr_m1 = vae_encode_fn(degraded_m1)
                encoded_m2, recon_psnr_m2 = vae_encode_fn(degraded_m2)
                encoded_gt, recon_psnr_gt = vae_encode_fn(gt)
                info['m1_psnr'] = recon_psnr_m1
                info['m2_psnr'] = recon_psnr_m2
                info['gt_psnr'] = recon_psnr_gt
                
            saved_dict = {
                "__key__": img_path_p1.stem,
                'info.json': json.dumps(info),
                'm1_clean.jpg': tensor_img_to_bytes(m1),
                'm2_clean.jpg': tensor_img_to_bytes(m2),
                'm1_degraded.jpg': tensor_img_to_bytes(degraded_m1),
                'm2_degraded.jpg': tensor_img_to_bytes(degraded_m2),
                'gt.jpg': tensor_img_to_bytes(gt),
            }
            if vae_encode_fn is not None:
                saved_dict['encoded_m1_degraded.npy'] = encoded_m1
                saved_dict['encoded_m2_degraded.npy'] = encoded_m2
                saved_dict['encoded_gt.npy'] = encoded_gt
            if llm_encode_fn is not None:
                saved_dict['encoded_instruction.npy'] = llm_encode_fn(instruction)
                
            sink.write(saved_dict)
            print(f'degradation: {info["degrations"]} to {img_path_p1.stem}')
        
            if n_sample_saved >= num_samples:
                print(f'generated {num_samples} image pairs')
                break
        
        
if __name__ == '__main__':
    from loguru import logger
    
    with logger.catch():
        pipe = make_degradation_pipe(1)

        modality1_dir = '/Data3/cao/ZiHanCao/datasets/VIF-MSRS/train/vi'
        modality2_dir = '/Data3/cao/ZiHanCao/datasets/VIF-MSRS/train/ir'
        gt_dir = '/Data3/cao/ZiHanCao/datasets/VIF-MSRS/train/gt'
        dataset_pattern = f'/Data3/cao/ZiHanCao/exps/panformer/task_datasets/VIF-MSRS/Degraded/%04d.tar'
        generate_degradation_instruction_pairs_from_dir(
            pipe, 
            modality1_dir, 
            modality2_dir, 
            gt_dir,
            dataset_pattern,
            pipe_apply_policy=None,
            vae_encode_fn=make_encode_fn(encode_type='lfq_vae', half_precision=False),
            llm_encode_fn=make_llm_encode_fn(llm_model='google-t5/t5-small'),
            num_samples=4000,
        )
        
        #* test pipe
        # img = read_image('/Data3/cao/ZiHanCao/datasets/VIF-MSRS/train/vi/00034N.jpg')
        # set_global_haze_depth_map_path('/Data3/cao/ZiHanCao/datasets/VIF-MSRS/train/vi/00034N.jpg')
        # m1 = pipe(img)
        # params = pipe._params
        # print(params)
        # m2 = pipe(img, params=params)
        
        # m3, m4 = pipe(img, img)
        # pass