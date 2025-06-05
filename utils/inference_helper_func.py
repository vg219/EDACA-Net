# GPL License
# Copyright (C) 2024 , UESTC

# All Rights Reserved
#
# @Time    : 2021/10/15 17:53
# @Author  : Zihan Cao, Xiao Wu


from functools import partial
import inspect
from typing import Tuple, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch import Tensor, nn
from tqdm import tqdm
import json
import re
import glob

from .visualize import viz_batch, res_image
from .metric_sharpening import AnalysisPanAcc
from utils.log_utils import easy_logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model.base_model import BaseModel
    
logger = easy_logger(func_name='inference_loop')


# signature finder
def has_patch_merge_model(model: "nn.Module | BaseModel"):
    return (hasattr(model, '_patch_merge_model')) or (hasattr(model, 'patch_merge_model'))

def patch_merge_in_val_step(model: "nn.Module | BaseModel", val_step_func: str='val_step'):
    assert hasattr(model, val_step_func), f'model should have {val_step_func} method'
    return 'patch_merge' in list(inspect.signature(getattr(model, val_step_func)).parameters.keys())


# callback function
def basic_callback(model: "BaseModel", iter_idx: int):
    from utils import get_local
    assert get_local().cache is not None and get_local.is_activate
    
    cache = get_local().cache
    attns = cache['_forward_implem']
    
    get_local.clear()
    
    
# dict data to device and type
def dict_data_to_device_and_type(data: dict, device: "str | torch.device | None", dtype: "torch.dtype | None"):
    if device is None:
        device = 'cpu'
    if dtype is None:
        dtype = torch.float32
        
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device=device, dtype=dtype)
        elif isinstance(v, dict):
            data[k] = dict_data_to_device_and_type(v, device, dtype)
        elif isinstance(v, list):
            data[k] = [dict_data_to_device_and_type(x, device, dtype) for x in v]
    
    return data
                


############ Main inference function ############

@torch.no_grad()
@torch.inference_mode()
def unref_for_loop(model,
                   dl: DataLoader,
                   device,
                   *,
                   split_patch=False,
                   feature_callback: callable=None,
                   **patch_merge_module_kwargs):
    from model.base_model import PatchMergeModule
    
    all_sr = []
    try:
        spa_size = tuple(dl.dataset.lms.shape[-2:])
    except AttributeError:
        spa_size = tuple(dl.dataset.rgb.shape[-2:])
    
    inference_bar = tqdm(enumerate(dl, 1), dynamic_ncols=True, total=len(dl))
    
    analysis = AnalysisPanAcc(ratio=patch_merge_module_kwargs.get('ergas_ratio', 4), ref=False,
                              sensor=patch_merge_module_kwargs.get('sensor', 'DEFAULT'),
                              default_max_value=patch_merge_module_kwargs.get('default_max_value', None))
    
    if split_patch:
        # check if has the patch merge model
        if not (has_patch_merge_model(model) or patch_merge_in_val_step(model)):
            # assert bs == 1, 'batch size should be 1'
            
            # warp the model into PatchMergeModule
            model = PatchMergeModule(net=model, device=device, **patch_merge_module_kwargs)
            
    for i, data in inference_bar:
        data = dict_data_to_device_and_type(data, device, dtype=torch.float32)
                
        # split the image into several patches to avoid gpu OOM
        if split_patch:
            if hasattr(model, 'forward_chop'):
                input = (data['ms'], data['lms'], data['pan'])
                sr = model.forward_chop(*input)[0]
            elif patch_merge_in_val_step(model):
                sr = model(**data, cfg=None, mode='sharpening_val', patch_merge=True)
            else:
                raise NotImplemented('model should have @forward_chop or patch_merge arg in @val_step')
        else:
            if patch_merge_in_val_step(model):
                sr = model(**data, cfg=None, mode='sharpening_val', patch_merge=False)
            else:
                sr = model(**data, cfg=None, mode='sharpening_val', patch_merge=False)
                
        sr = sr.clip(0, 1)
        sr1 = sr.detach().cpu().numpy()
        all_sr.append(sr1)
        
        # analysis(sr, ms, lms, pan)
        
        viz_batch(sr.detach().cpu(), suffix='sr', start_index=i, base_path='visualized_img/img_shows')
        viz_batch(ms.detach().cpu(), suffix='ms', start_index=i, base_path='visualized_img/img_shows')
        viz_batch(pan.detach().cpu(), suffix='pan', start_index=i, base_path='visualized_img/img_shows')
        
        if feature_callback is not None:
            feature_callback(model, i)
        
    logger.info(analysis.print_str())

    return all_sr


@torch.no_grad()
@torch.inference_mode()
def ref_for_loop(model: "BaseModel",
                 dl: DataLoader,
                 device: "str | torch.device | None",
                 *,
                 split_patch=False,
                 ergas_ratio=4,
                 residual_exaggerate_ratio=100,
                 feature_callback: callable=None,
                 **patch_merge_module_kwargs):
    from model.base_model import PatchMergeModule
    
    analysis = AnalysisPanAcc(ergas_ratio)
    all_sr = []
    inference_bar = tqdm(enumerate(dl, 1), dynamic_ncols=True, total=len(dl))
    if device is None:
        device = next(iter(model.parameters())).device
    else:
        assert next(iter(model.parameters())).device == torch.device(device), 'model device should be the same as input device'

    if not (has_patch_merge_model(model) or patch_merge_in_val_step(model)):
            # assert bs == 1, 'batch size should be 1'
            
            # warp the model into PatchMergeModule
            model = PatchMergeModule(net=model, device=device, **patch_merge_module_kwargs)
    for i, data in inference_bar:
        data = dict_data_to_device_and_type(data, device, dtype=torch.float32)
        gt = data['gt']
        
        # split the image into several patches to avoid gpu OOM
        if split_patch:
            if hasattr(model, 'forward_chop'):
                input = (data['ms'], data['lms'], data['pan'])
                sr = model.forward_chop(*input)[0]
            elif patch_merge_in_val_step(model):
                # TODO: cfg is set to None, the data keys should be same with the signature of `val_step` or `sharpening_val_step`
                sr = model(**data, cfg=None, mode='sharpening_val')
            else:
                raise NotImplemented('model should have @forward_chop or patch_merge arg in @val_step')
        # no split patch
        else:
            if patch_merge_in_val_step(model):
                # has patch merge model in model registered, but we do not use it
                sr = model(**data, cfg=None, mode='sharpening_val', patch_merge=False)
            else:
                sr = model(**data, cfg=None, mode='sharpening_val', patch_merge=False)
                
        if feature_callback is not None:
            feature_callback(model, i)
                
        # cache = get_local().cache
        # attns = cache['FirstAttn.forward']
        
        ## panRWKV output feature
        # if i in [4]:
        #     cache = get_local().cache
        #     out = cache['RWKVBlock_v2.forward']
            
        #     torch.save(out, f'visualized_img/lformer_full_attn_feat/output_feat_{i}.pth')

        # get_local.clear()
        
        ## save panMamba updated_xs
        # if i in [11]:
        #     cache = get_local().cache
        #     # feat_ssm_states = cache['UniSequential.LEMM_enc_forward']
        #     # attns = cache['MSReversibleRefine.forward']
        #     feat_updated_xs = cache['cross_selective_scan']
        #     # for x in attns:
        #     #     if x[0] is not None:
        #     #         x[0] = x[0].to(torch.float16)
        #             # x[1] = x[1].to(torch.float16)
                    
        #     for x in feat_updated_xs:
        #         if x[0] is not None:
        #             x[0] = x[0].to(torch.float16)
        #         if x[1] is not None:
        #             x[1] = x[1].to(torch.float16)
                    
        #     torch.save(feat_updated_xs, f'/Data2/ZiHanCao/exps/panformer/visualized_img/updated_xs/updated_xs_wv3_{i}.pth')
        # print('saved pth file...')
        # get_local.clear()
                
        sr = sr.clip(0, 1)
        sr1 = sr.detach().cpu().numpy()
        all_sr.append(sr1)

        analysis(gt, sr)

        res = res_image(gt, sr, exaggerate_ratio=residual_exaggerate_ratio)
        viz_batch(sr.detach().cpu(), suffix='sr', start_index=i, base_path='visualized_img/img_shows')
        viz_batch(gt.detach().cpu(), suffix='gt', start_index=i, base_path='visualized_img/img_shows')
        viz_batch(ms.detach().cpu(), suffix='ms', start_index=i, base_path='visualized_img/img_shows')
        viz_batch(pan.detach().cpu(), suffix='pan', start_index=i, base_path='visualized_img/img_shows')
        viz_batch(res.detach().cpu(), suffix='residual', start_index=i, base_path='visualized_img/img_shows')

        # print(f'PSNR: {psnr}, SSIM: {ssim}')

    print(analysis.print_str())

    return all_sr


def find_data_path(dataset_type, full_res):
    if dataset_type == "wv3":
        if not full_res:
            path = "/volsparse1/dataset/PanCollection/test_data/test_wv3_multiExm1.h5"
        else:
            # path = '/home/ZiHanCao/datasets/pansharpening/wv3/full_examples/test_wv3_OrigScale_multiExm1.h5'
            path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_wv3_OrigScale_multiExm1.h5"
    elif dataset_type == "cave":
        path = "/Data2/ZiHanCao/datasets/HISI/new_cave/test_cave(with_up)x4.h5"
    elif dataset_type == "cave_x8":
        path = "/volsparse1/dataset/HISR/cave_x8/test_cave(with_up)x8_rgb.h5"
    elif dataset_type == "harvard":
        # path = "/Data2/ZiHanCao/datasets/HISI/new_harvard/test_harvard(with_up)x4_rgb.h5"
        path = "/Data2/ShangqiDeng/data/HSI/harvard_x4/test_harvard(with_up)x4_rgb200.h5"
    elif dataset_type == "harvard_x8":
        path = "/volsparse1/dataset/HISR/harvard_x8/test_harvard(with_up)x8_rgb.h5"
    elif dataset_type == "gf5":
        if not full_res:
            path = "/Data2/ZiHanCao/datasets/pansharpening/GF5-GF1/tap23/test_GF5_GF1_23tap_new.h5"
        else:
            path = "/Data2/ZiHanCao/datasets/pansharpening/GF5-GF1/tap23/test_GF5_GF1_OrigScale.h5"
    elif dataset_type == "gf":
        if not full_res:
            path = "/Data2/ZiHanCao/datasets/pansharpening/gf/reduced_examples/test_gf2_multiExm1.h5"
        else:
            # path = '/home/ZiHanCao/datasets/pansharpening/gf/full_examples/test_gf2_OrigScale_multiExm1.h5'
            path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_gf2_OrigScale_multiExm1.h5"
    elif dataset_type == "qb":
        if not full_res:
            path = "/Data2/ZiHanCao/datasets/pansharpening/qb/reduced_examples/test_qb_multiExm1.h5"
        else:
            # path = '/home/ZiHanCao/datasets/pansharpening/qb/full_examples/test_qb_OrigScale_multiExm1.h5'
            path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_qb_OrigScale_multiExm1.h5"
    elif dataset_type == "wv2":
        if not full_res:
            path = "/Data2/ZiHanCao/datasets/pansharpening/wv2/reduced_examples/test_wv2_multiExm1.h5"
        else:
            # path = '/home/ZiHanCao/datasets/pansharpening/wv2/full_examples/test_wv2_OrigScale_multiExm1.h5'
            path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_wv2_OrigScale_multiExm1.h5"
    elif dataset_type == "roadscene":
        path = "/Data2/ZiHanCao/datasets/RoadSceneFusion_1"
    elif dataset_type == "tno":
        path = "/Data2/ZiHanCao/datasets/TNO"
    else:
        raise NotImplementedError("not exists {} dataset".format(dataset_type))

    return path

def find_key_args_in_log(arch, sub_arch, datasets, weight_path):
    # handle weight_path
    slash_with_id = re.findall(r'_[a-zA-Z0-9-]{8}(?=\.pth|_)', weight_path)[-1]
    run_id = slash_with_id[1:]
    
    if sub_arch is not None and sub_arch != '': 
        sub_arch = '_' + sub_arch
    else:
        sub_arch = ''
    _log_path = f'log_file/{arch}{sub_arch}/{datasets}/*{run_id}*/config.json'
    log_path = glob.glob(_log_path)
    if len(log_path) != 1:
        raise RuntimeError(f'>>> log file: {_log_path} not exists!')
    print(f'>>> found run id: {log_path[0]} config')
    args = json.loads(''.join(open(log_path[0], 'r').readlines()))
    
    return args
    


def crop_inference(model: "BaseModel",
                   xs: Tuple[Tensor, Tensor, Tensor],
                   crop_size: Tuple[int] = (16, 64, 64),
                   stride: Tuple[int] = (8, 32, 32)):
    # only support CAVE dataset
    # input shape: 128, 512, 512

    # xs: (hsi_lr, hsi_up, rgb)

    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True

    # preprocessing
    crop_xs = []
    ncols = []
    bs, out_c, _, _ = xs[0].shape
    _, _, out_h, out_w = xs[-1].shape
    for i in range(len(xs)):
        x = xs[i]
        _, c, h, _ = x.shape  # assume h equals w
        crop = crop_size[i]
        s = stride[i]

        ncol = (h - crop) // s
        ncols.append(ncol)
        crop_x = F.unfold(x, crop, stride=s)
        crop_x = einops.rearrange(crop_x, 'b (c k l) m -> m b c k l', k=crop, l=crop, c=c)
        crop_xs.append(crop_x)

    # model inference
    model.eval()
    out = []
    for i in range(crop_xs[0].size(0)):
        input = [crop_xs[j][i].cuda(0) for j in range(len(xs))]
        out.append(model.val_step(*input).detach().cpu())  # [bs * 225, 31, 64, 64]
        del input
        torch.cuda.empty_cache()
    # input: 255*[b, 31, 64, 64]
    out = torch.cat(out, dim=0)

    # postprocessing
    out = einops.rearrange(out, '(m b) c k l -> b (c k l) m', b=bs, k=crop_size[-1], l=crop_size[-1], c=out_c)
    output = F.fold(out, output_size=(out_h, out_w),
                    kernel_size=(crop_size[-1], crop_size[-1]),
                    dilation=1,
                    padding=0,
                    stride=(stride[-1], stride[-1]))

    # ncol = ncols[-1]
    # out = out.view(bs, -1, out_c, crop_size[-2], crop_size[-1])  # [bs, 225, 64, 64]
    # output = torch.zeros(bs, out_c, out_h, out_w)
    # for bi in range(bs):
    #     for i in range(ncol):
    #         for j in range(ncol):
    #             y = out[bi]  # [255, 64, 64]

    return output


if __name__ == '__main__':
    from model.dcformer_reduce import DCFormer_Reduce

    model = DCFormer_Reduce(8, 'C').cuda(0)

    ms = torch.randn(1, 8, 128, 128)
    interp_ms = F.interpolate(ms, size=512)

    lms = torch.randn(1, 8, 512, 512)
    pan = torch.randn(1, 1, 512, 512)
    expand_pan = pan.expand(-1, 8, -1, -1)

    # print(model.val_step(ms, lms, pan).shape)

    print(crop_inference(model, xs=(ms, lms, pan)).shape)
