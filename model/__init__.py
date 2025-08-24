import os
import sys
import importlib
from importlib.util import find_spec, LazyLoader, module_from_spec
sys.path.append(os.path.dirname(__file__))

## suppress warnings
import warnings
warnings.filterwarnings("ignore", module="torch.utils")
warnings.filterwarnings("ignore", module="deepspeed.accelerator")

## register all models
from model.base_model import MODELS, BaseModel
from utils import easy_logger

logger = easy_logger(func_name='model_registry')

__all__ = [
    'MODELS',
    'BaseModel',
    'build_network',
]

_all_modules = [
    'panrwkv_v8_cond_norm',
    'panRWKV_v9_local',
    'RWKVFusion_v10_multi_modal',
    'RWKVFusion_v12',
    'MHIIF',
    'MHIIF_g',
    'FeINFN',
    'hermite',
    'MHIIF_J',
    'MHIIF_J2',
    'MHIIF_rbf',
    'ENACIR',
    'ENACIR_V2',
    'MHIIF_J2_Hermite',
    'MIMO_SST',
    'DHIF',
    'DCT',
]

_all_model_class_name = [
    'RWKVFusion',
    'RWKVFusion',
    'RWKVFusion',
    'RWKVFusion',
    'MHIIF_',
    'MHIIF_g',
    'FeINFNet',
    'hermite',
    'MHIIF_J',
    'MHIIF_J2',
    'MHIIF_rbf',
    'ENACIR',
    'ENACIR_V2',
    'MHIIF_J2_Hermite',
    'Net',
    'HSI_Fusion',
    'DCT',
]

assert len(_all_modules) == len(_all_model_class_name), 'length of modules and registry names should be the same'

_module_network_dict = {k: v for k, v in zip(_all_modules, _all_model_class_name)}


def _lazy_load_module():
    lazy_loader_dict = {}
    for module_name, class_name in _module_network_dict.items():
        spec = find_spec(module_name)
        if spec is not None:
            spec.loader = LazyLoader(spec.loader)
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            lazy_loader_dict[f'{module_name}.{class_name}'] = module
        else:
            raise ValueError(f'Module: {module_name} not found')
    return lazy_loader_dict
            
LAZY_LOADER_DICT = _lazy_load_module()

def _active_load_module(module_name, model_class_name):
    dict_name = f'{module_name}.{model_class_name}'
    if dict_name in LAZY_LOADER_DICT:
        logger.info(f'loading {dict_name}')
        module = LAZY_LOADER_DICT[dict_name]
        return getattr(module, model_class_name)
    else:
        logger.critical(f'{dict_name} not found in LAZY_LOADER_DICT')
        raise ValueError(f'{dict_name} not found in LAZY_LOADER_DICT')


# TODO: hydra import
# ==============================================
# register all models
# from model.DCFNet import DCFNet
# from model.FusionNet import FusionNet
# from model.PANNet import VanillaPANNet

# from model.M3DNet import M3DNet
# from model.panformer import PanFormerGAU, PanFormerUNet2, PanFormerSwitch, PanFormerUNet, PanFormer
# from model.dcformer import DCFormer
# from model.dcformer_dpw import DCFormer_DPW
# from model.dcformer_dpw_woo import DCFormer_DPW_WOO
# from model.dcformer_dynamic import DCFormerDynamicConv
# from model.dcformer_reduce import DCFormer_Reduce
# from model.dcformer_mwsa import DCFormerMWSA
# from model.dcformer_mwsa_wx import DCFormerMWSA

# from model.fuseformer import MainNet

# from model.dcformer_reduce_c_64 import DCFormer_Reduce_C64
# from model.dcformer_reduce_c_32_tmp import DCFormer_Reduce_C32
# from model.dcformer_sg_c32 import DCFormer_SG_C32
# from model.dcformer_mobile_x8 import DCFormerMobile
# from model.ydtr import MODEL as YDTR
# from model.CSSNet import Our_netf
# from model.mmnet import MMNet
# from model.pmacnet import PMACNet
# from model.SSRNet import SSRNET
# from model.hsrnet import HSRNet
# from model.restfnet import ResTFNet
# from model.HPMNet import fusionnet

# ablation
# from model.dcformer_abla_only_channel_attn import DCFormer_XCA
# from model.dcformer_abla_only_mwa import DCFormerOnlyMWA
# from model.dcformer_abla_only_cross_branch_mwsa import DCFormerOnlyCrossBranchMWSA
# from model.ablation_exps.dcformer_abla_wo_ghost_module import DCFormerMWSA
# from model.ablation_exps.dcformer_abla_only_XCA import DCFormerMWSA
# from model.ablation_exps.dcformer_abla_only_MWSA import DCFormerMWSA
# from model.ablation_exps.dcformer_abla_in_scale_MWSA import DCFormerMWSA

# disscussion
# from model.dcformer_disscuss_mog_fusion_head import DCFormerMWSAMoGFusionHead
# from model.dcformer_dissucss_multisource_proj import DCFormerMWSAMultiSourceProj

# from model.LFormer import AttnFuseMain
# from model.lformer_reduced_swin_attn import AttnFuseMain
# from model.lformer_ablation.LFormer_ablation_skip_attn import AttnFuseMain

# from model.reciprocal_transformer import DCT

# from model.panMamba_old import ConditionalNAFNet
# from model.panMamba import ConditionalNAFNet
# from model.panMamba_ablation.panMamba_only_conv_NAF import ConditionalNAFNet

# from model.panRWKV_v2 import ConditionalNAFNet
# from model.panRWKV_v3 import ConditionalNAFNet

# from model.panRWKV_v4 import RWKVFusion

# from model.panRWKV_v5_cross import RWKVCrossFusion
# from model.panrwkv_v7_k1 import RWKVFusion
    
# from model.panrwkv_v8_cond_norm import RWKVFusion

# from model.MGDN import MGFF


# from model.MIMO_SST import Net
# from model.panmamba_zhouman import Net

# from model.panRWKV import ConditionalNAFNet

# others
# from model.GPPNN import GPPNN

# ============================================== model registry ==================================

# FIXME: may cause conficts with other arguments in args that rely on static registered model name in main.py
def import_model_from_name(name):
    module = importlib.import_module(name, package='model')
    model_cls = getattr(module, name)
    return model_cls

def build_network(model_name: str=None, **kwargs) -> BaseModel:
    """
    build network from model name and kwargs
    
    """
    
    assert model_name is not None, 'model_name is not specified'
    try:
        net = MODELS.get(model_name)
    except:
        try:
            net = import_model_from_name(model_name)
        except:
            net = MODELS.get(model_name.split('.')[-1])
        
    assert net is not None, f'no model named {model_name} is registered'
    
    return net(**kwargs)

def lazy_load_network(module_name: str=None,
                      model_class_name:str=None, **kwargs) -> BaseModel:
    """
    lazy load model from module_name and registry_model_name
    
    """
    
    assert module_name is not None, 'module_name is not specified'
    assert model_class_name is not None,'model_class_name is not specified'
    
    return _active_load_module(module_name, model_class_name)(**kwargs)



## hydra import

import hydra
from omegaconf import OmegaConf

def build_model_from_name(cfg: OmegaConf):
    model = hydra.utils.instantiate(cfg)
    
    return model

# ==============================================


