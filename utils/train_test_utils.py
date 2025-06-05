from io import BytesIO
import zipfile
from pathlib import Path
from PIL import Image
import PIL.Image as Image
from contextlib import contextmanager
from collections import OrderedDict
from typing import Union
from omegaconf import OmegaConf, DictConfig

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from utils import (h5py_to_dict,
                   NameSpace,
                   easy_logger,
                   default_none_getattr,
                   default_getattr,
                   dict_to_namespace,
                   catch_any_error,
                   merge_args)
from utils.cfg_utils import omegaconf_create

FUSION_TASK_DATASETS_MAPPING = {
    'roadscene': 'VIF',
    'tno': 'VIF',
    'msrs': 'VIF',
    'llvip': 'VIF',
    'm3fd': 'VIF',
    'med_harvard': 'medical_fusion',
    'sice': 'MEF',
    'mefb': 'MEF',
    'realmff': 'MFF',
    'mff_whu': 'MFF',
    'lytro': 'MFF',
    'wv3': 'Pansharpening',
    'qb': 'Pansharpening',
    'gf2': 'Pansharpening',
    'hisi-houston': 'Pansharpening',
    'cave_x4': 'HMIF',
    'harvard_x4': 'HMIF',
    'cave_x8': 'HMIF',
    'harvard_x8': 'HMIF',
    'pavia': 'HMIF',
    'chikusei': 'HMIF',
    'botswana': 'HMIF',
    # ========= Degradation fusion task ==========
    'dif': 'DIF',
}

ALL_TASKS = [
    'Pansharpening',
    'HMIF',
    'VIF',
    'MEF',
    'MFF',
    'medical_fusion',
    'DIF',
]


def get_eval_dataset(args: NameSpace,
                     logger=None):
    from task_datasets import (
        WV3Datasets, 
        GF2Datasets, 
        HISRDatasets, 
        TNODataset, 
        RoadSceneDataset, 
        LLVIPDALIPipeLoader, 
        MSRSDatasets, 
        M3FDDALIPipeLoader, 
        MedHarvardDataset, 
        SICEDataset,
        RealMFFDataset,
        MFFWHUDataset,
        MEFBDataset,
        LytroDataset,
    )

    logger = easy_logger(func_name='get_eval_dataset')

    val_ds, val_dl = None, None

    logger.info(f"use dataset: {args.dataset} on fusion task")
    args.fusion_task = FUSION_TASK_DATASETS_MAPPING.get(args.dataset, 'UNSUPPORTED')
    if args.fusion_task == 'UNSUPPORTED':
        raise ValueError(f"not support dataset {args.dataset}")
    logger.info(f"dataset {args.dataset} on fusion task {args.fusion_task}")
    
    # 1. vis-ir image fusion (without gt)
    if args.dataset == "roadscene":
        val_ds = RoadSceneDataset(
            args.path.base_dir, 
            "test",
            no_split=True,
            get_name=True,
            only_resize=None,
            output_none_mask=True,
            with_txt_feature=default_getattr(args, 'datasets_cfg.roadscene.with_txt_feature', False),
        )
    elif args.dataset == "tno":
        val_ds = TNODataset(
            args.path.base_dir,
            "test",
            aug_prob=0.0,
            no_split=True,
            get_name=True,
            output_none_mask=False,
            with_txt_feature=default_getattr(args, 'datasets_cfg.tno.with_txt_feature', False),
            only_resize=None,
        )
    elif args.dataset == "msrs":
        val_ds = MSRSDatasets(
            args.path.base_dir,
            mode=args.dataset_mode,  # or 'test'/'detection'
            transform_ratio=0.0,
            get_name=True,
            reduce_label=default_getattr(args, 'datasets_cfg.msrs.reduce_label', True),
            with_txt_feature=default_getattr(args, 'datasets_cfg.msrs.with_txt_feature', False),
            with_mask=default_getattr(args, 'datasets_cfg.msrs.with_mask', False)
        )
    elif args.dataset == "llvip":
        args.fusion_task = 'VIF'
        val_dl = LLVIPDALIPipeLoader(
            args.path.base_dir,
            "test",
            batch_size=args.val_bs,
            device=args.device,
            shuffle=False,
            with_mask=default_getattr(args, 'datasets_cfg.llvip.with_mask', False),
            reduce_label=default_getattr(args, 'datasets_cfg.llvip.reduce_label', True),
            with_txt_feature=default_getattr(args, 'datasets_cfg.llvip.with_txt_feature', False),
            get_name=True,
        )
    elif args.dataset == "m3fd":
        val_dl = M3FDDALIPipeLoader(
            args.path.base_dir,
            "test",
            batch_size=args.val_bs,
            device=args.device,
            shuffle=False,
            with_mask=default_getattr(args, 'datasets_cfg.m3fd.with_mask', False),
            reduce_label=default_getattr(args, 'datasets_cfg.m3fd.reduce_label', True),
            with_txt_feature=default_getattr(args, 'datasets_cfg.m3fd.with_txt_feature', False),
            get_name=True,
        )

    elif args.dataset == "med_harvard":
        val_ds = MedHarvardDataset(
            args.path.base_dir,
            mode="test",
            device=args.device,
            data_source="xmu",
            get_name=True,
            task=default_getattr(args, 'datasets_cfg.med_harvard.task', None),
            with_mask=default_getattr(args, 'datasets_cfg.med_harvard.with_mask', False),
            reduce_label=default_getattr(args, 'datasets_cfg.med_harvard.reduce_label', True),
            with_txt_feature=default_getattr(args, 'datasets_cfg.med_harvard.with_txt_feature', False),
            only_resize=default_getattr(args, 'datasets_cfg.med_harvard.only_resize', None),
        )
    elif args.dataset == "sice":
        val_ds = SICEDataset(
            data_dir=args.path.base_dir,
            mode="test",
            transform_ratio=0.0,
            with_mask=default_getattr(args, 'datasets_cfg.sice.with_mask', False),
            with_txt=default_getattr(args, 'datasets_cfg.sice.with_txt', False),
            only_y=default_getattr(args, 'datasets_cfg.sice.only_y', False),
            get_name=True,
        )
    elif args.dataset == "mefb":
        val_ds = MEFBDataset(
            data_dir=args.path.base_dir,
            mode="test",
            transform_ratio=0.0,
            with_mask=default_getattr(args, 'datasets_cfg.mefb.with_mask', False),
            with_txt=default_getattr(args, 'datasets_cfg.mefb.with_txt', False),
            only_y=default_getattr(args, 'datasets_cfg.mefb.only_y', False),
            get_name=True,
        )
    elif args.dataset == "realmff":
        val_ds = RealMFFDataset(
            data_dir=args.path.base_dir,
            mode="test",
            transform_ratio=0.0,
            with_mask=default_getattr(args, 'datasets_cfg.realmff.with_mask', False),
            with_txt=default_getattr(args, 'datasets_cfg.realmff.with_txt', False),
            get_name=True,
        )
    elif args.dataset == "mff_whu":
        val_ds = MFFWHUDataset(
            data_dir=args.path.base_dir,
            mode="test",
            transform_ratio=0.0,
            with_mask=default_getattr(args, 'datasets_cfg.mff_whu.with_mask', False),
            with_txt=default_getattr(args, 'datasets_cfg.mff_whu.with_txt', False),
            get_name=True,
        )
    elif args.dataset == "lytro":
        val_ds = LytroDataset(
            data_dir=args.path.base_dir,
            with_mask=default_getattr(args, 'datasets_cfg.lytro.with_mask', False),
            with_txt=default_getattr(args, 'datasets_cfg.lytro.with_txt', False),
            get_name=True,
        )

    ## 2. sharpening datasets (with gt)
    elif args.dataset in [
        "wv3",
        "qb",
        "gf2",
        "cave_x4",
        "harvard_x4",
        "cave_x8",
        "harvard_x8",
        "hisi-houston",
        "pavia",
        "chikusei",
        "botswana",
    ]:
        args.fusion_task = 'Pansharpening'
        
        logger.info(f"use dataset: {args.dataset} on pansharpening/HISR task")
        # FIXME: 需要兼顾老代码（只有trian_path和val_path）的情况
        if hasattr(args.path, "val_path"):
            # 旧代码：手动切换数据集路径
            val_path = args.path.val_path
        else:
            if not args.full_res:
                val_path = default_none_getattr(args, f'path.{args.dataset}_val_path')
            else:
                val_path = default_none_getattr(args, f'path.{args.dataset}_full_path')
            
        assert val_path is not None, "val_path should not be None"

        if val_path is not None:
            assert val_path.endswith(".h5"), 'val_path should end with ".h5"'

        h5_val = h5py.File(val_path)

        # 1. parsharpening
        if args.dataset in ["wv3", "qb"]:
            d_val = h5py_to_dict(h5_val)
            val_ds = WV3Datasets(d_val, 
                                 hp=default_getattr(args, f'datasets_cfg.{args.dataset}.hp', False),
                                 full_res=default_getattr(args, 'full_res', False),
                                 aug_prob=0.0,
                                 txt_file=default_none_getattr(args, f'path.{args.dataset}_txt_full_path'))
        elif args.dataset == "gf2":
            d_val = h5py_to_dict(h5_val)
            val_ds = GF2Datasets(d_val, 
                                 hp=default_getattr(args, 'datasets_cfg.gf2.hp', False),
                                 full_res=default_getattr(args, 'full_res', False),
                                 aug_prob=0.0,
                                 txt_file=default_none_getattr(args, 'path.gf2_txt_val_path'),
                                 txt_feature_online_load=True)  # txt feature is too large to load into memory

        # 2. hyperspectral image fusion
        elif (args.dataset[:4] == "cave" or 
              args.dataset[:7] == "harvard" or
              args.dataset[:8] == "chikusei" or
              args.dataset[:5] == "pavia" or
              args.dataset[:8] == "botswana" or
              args.dataset[:7] == "houston"
            ):
            args.fusion_task = 'HMIF'
            
            if args.dataset in ["pavia", "botswana"]:
                keys = ['GT', 'MS', 'PAN', 'LMS']
            else:
                keys = ["LRHSI", "HSI_up", "RGB", "GT"]
                
            if args.dataset.split("-")[-1] == "houston":
                from einops import rearrange
                # to avoid unpicklable error for clourse in class
                def permute_fn(x):
                    return rearrange(x, "b h w c -> b c h w")
                dataset_fn = permute_fn
            else:
                dataset_fn = None

            d_val = h5py_to_dict(h5_val, keys)
            dataset_txt_paths = {
                "cave": "path.cave_txt_val_path",
                "harvard": "path.harvard_txt_val_path",
                "chikusei": "path.chikusei_txt_val_path",
                "pavia": "path.pavia_txt_val_path",
                "botswana": "path.botswana_txt_val_path",
                "houston": "path.houston_txt_val_path",
            }
        
            txt_file_path = dataset_txt_paths.get(args.dataset, None)
            txt_file = default_none_getattr(args, txt_file_path)

            val_ds = HISRDatasets(
                d_val, 
                txt_file=txt_file,
                dataset_fn=dataset_fn,
                dataset_name=args.dataset,
            )
            
    else:
        raise NotImplementedError(f"not support dataset {args.dataset}")

    return val_ds, val_dl


def get_fusion_dataset(args: "NameSpace | DictConfig"):
    from accelerate import PartialState
    
    state = PartialState()
    device = state.device
    
    train_ds, val_ds, train_dl, val_dl = None, None, None, None

    if args.dataset in [
        "flir",
        "tno",
        "roadscene_tno_joint",
        "vis_ir_joint",
        "msrs",
        "llvip",
        "med_harvard",
        "m3fd",
        "sice",
        "mefb",
        "realmff",
        "mff_whu",
    ]:
        args.task = "fusion"
        args.path.base_dir = getattr(args.path, f"{args.dataset}_base_dir")
        
        if args.dataset == "roadscene":
            args.fusion_task = 'VIF'
            from task_datasets import RoadSceneDataset

            train_ds = RoadSceneDataset(args.path.base_dir, "train")
            val_ds = RoadSceneDataset(args.path.base_dir, "test")
            
        elif args.dataset in ["tno", "roadscene_tno_joint"]:
            from task_datasets import TNODataset
            
            args.fusion_task = 'VIF'
            train_ds = TNODataset(
                args.path.base_dir,
                "train",
                aug_prob=args.aug_probs[0],
                duplicate_vis_channel=True,
            )
            val_ds =  Dataset(
                args.path.base_dir,
                "test",
                aug_prob=args.aug_probs[1],
                no_split=True,
                duplicate_vis_channel=True,
            )
            
        elif args.dataset == "msrs":
            from task_datasets.VIF.MSRS import MSRSDatasets
            ds_kwargs = {
                'reduce_label': default_getattr(args, 'datasets_cfg.msrs.reduce_label', True),
                'only_resize': default_getattr(args, 'datasets_cfg.msrs.only_resize', None),
                'with_txt_feature': default_getattr(args, 'datasets_cfg.msrs.with_txt_feature', False),
                'with_mask': default_getattr(args, 'datasets_cfg.msrs.with_mask', False),
            }

            args.fusion_task = 'VIF'
            train_ds = MSRSDatasets(
                args.path.base_dir,
                "train",
                transform_ratio=default_getattr(args, 'datasets_cfg.msrs.transform_ratio', args.aug_probs[0]),
                output_size=default_getattr(args, 'datasets_cfg.msrs.output_size', None),
                n_proc_load=1,
                only_y_component=False,
                **ds_kwargs,
            )
            val_ds = MSRSDatasets(
                args.path.base_dir,
                "test",
                transform_ratio=default_getattr(args, 'datasets_cfg.msrs.transform_ratio', args.aug_probs[1]),
                output_size=default_getattr(args, 'datasets_cfg.msrs.output_size', args.fusion_crop_size),
                fast_eval_n_samples=default_getattr(args, 'datasets_cfg.msrs.fast_eval_n_samples', 80),
                n_proc_load=1,
                only_y_component=False,
                **ds_kwargs,
            )
            
        elif args.dataset == "llvip":
            from task_datasets import LLVIPDALIPipeLoader
            ds_kwargs = {
                'reduce_label': default_getattr(args, 'datasets_cfg.llvip.reduce_label', True),
                'only_resize': default_getattr(args, 'datasets_cfg.llvip.only_resize', None),
                'with_txt_feature': default_getattr(args, 'datasets_cfg.llvip.with_txt_feature', False),
                'with_mask': default_getattr(args, 'datasets_cfg.llvip.with_mask', False),
                'crop_strategy': default_getattr(args, 'datasets_cfg.llvip.crop_strategy', 'crop_resize'),
            }

            args.fusion_task = 'VIF'
            # We use DALI pipeline to accelerate the data loading process
            train_dl = LLVIPDALIPipeLoader(
                args.path.base_dir,
                "train",
                batch_size=args.train_bs,
                output_size=args.fusion_crop_size,
                device=state.device,
                num_shards=state.num_processes,
                shard_id=state.process_index,
                shuffle=True,
                only_y_component=False,
                **ds_kwargs,
            )
            val_dl = LLVIPDALIPipeLoader(
                args.path.base_dir,
                "test",
                batch_size=args.val_bs,
                device=state.device,
                fast_eval_n_samples=default_getattr(args, 'datasets_cfg.llvip.fast_eval_n_samples', 80),
                num_shards=state.num_processes,
                shard_id=state.process_index,
                shuffle=False,
                only_y_component=False,
                **ds_kwargs,
            )
            
        elif args.dataset == "m3fd":
            from task_datasets import M3FDDALIPipeLoader
            ds_kwargs = {
                'reduce_label': default_getattr(args, 'datasets_cfg.m3fd.reduce_label', True),
                'crop_strategy': default_getattr(args, 'datasets_cfg.m3fd.crop_strategy', 'crop_resize'),
                'with_txt_feature': default_getattr(args, 'datasets_cfg.m3fd.with_txt_feature', False),
                'output_size': default_getattr(args, 'datasets_cfg.m3fd.output_size', args.fusion_crop_size),
                'only_resize': default_getattr(args, 'datasets_cfg.m3fd.only_resize', None),
                'with_mask': default_getattr(args, 'datasets_cfg.m3fd.with_mask', False),
            }

            args.fusion_task = 'VIF'
            train_dl = M3FDDALIPipeLoader(
                args.path.base_dir,
                "train",
                batch_size=args.train_bs,
                device=state.device,
                num_shards=state.num_processes,
                shard_id=state.process_index,
                shuffle=True,
                only_y_component=False,
                **ds_kwargs,
            )
            val_dl = M3FDDALIPipeLoader(
                args.path.base_dir,
                "test",
                batch_size=args.val_bs,
                device=state.device,
                num_shards=state.num_processes,
                shard_id=state.process_index,
                shuffle=True,
                only_y_component=False,
                fast_eval_n_samples=default_getattr(args, 'datasets_cfg.m3fd.fast_eval_n_samples', 80),
                **ds_kwargs,
            )

        elif args.dataset == "vis_ir_joint":
            from task_datasets import VISIRJointGenericLoader
            additional_args = {
                'reduce_label': default_getattr(args, 'datasets_cfg.reduce_label', True),
                'crop_strategy': default_getattr(args, "datasets_cfg.vis_ir_joint.crop_strategy", 'crop_resize'),
                'with_txt_feature': default_getattr(args, 'datasets_cfg.vis_ir_joint.with_txt_feature', False),
                'output_size': default_getattr(args, 'datasets_cfg.vis_ir_joint.output_size', None),
                'only_resize': default_getattr(args, 'datasets_cfg.vis_ir_joint.only_resize', None),
                'with_mask': default_getattr(args, 'datasets_cfg.vis_ir_joint.with_mask', True),
            }

            args.fusion_task = 'VIF'
            train_dl = VISIRJointGenericLoader(
                vars(args.path.base_dir) if isinstance(args.path.base_dir, NameSpace) else
                OmegaConf.to_container(args.path.base_dir),
                mode="train",
                batch_size=args.train_bs,
                device=state.device,
                shuffle_in_dataset=True,
                num_shards=state.num_processes,
                shard_id=state.process_index,
                only_y_component=False,
                **additional_args
            )
            val_dl = VISIRJointGenericLoader(
                ## only test msrs and roadscene_tno_joint dataset
                {'msrs': args.path.base_dir['msrs'],
                 'roadscene_tno_joint': args.path.base_dir['roadscene_tno_joint']},
                mode="test",
                output_size=None,                # enforce the different size images to be the same size or batch size to be 1
                batch_size=args.val_bs,
                device=state.device,
                num_shards=state.num_processes,
                shard_id=state.process_index,
                only_y_component=False,
                crop_strategy=default_getattr(args, 'datasets_cfg.vis_ir_joint.crop_strategy', 'crop_resize'),
                random_datasets_getitem=default_getattr(args, 'datasets_cfg.vis_ir_joint.random_datasets_getitem', False),
                shuffle_in_dataset=True,        # ensure different images in the same batch for tensorboard visualization
                reduce_label=default_getattr(args, 'datasets_cfg.reduce_label', True),
                fast_eval_n_samples=default_getattr(args, 'datasets_cfg.vis_ir_joint.fast_eval_n_samples', 80),
                with_txt_feature=default_getattr(args, 'datasets_cfg.vis_ir_joint.with_txt_feature', False),
                only_resize=default_getattr(args, 'datasets_cfg.vis_ir_joint.only_resize', None),
            )

        elif args.dataset == "med_harvard":
            from task_datasets import MedHarvardDataset
            
            task = default_none_getattr(args, 'datasets_cfg.med_harvard.task')
            additional_args = {
                'with_mask': default_getattr(args, 'datasets_cfg.med_harvard.with_mask', False),
                'reduce_label': default_getattr(args, 'datasets_cfg.med_harvard.reduce_label', True),
                'with_txt_feature': default_getattr(args, 'datasets_cfg.med_harvard.with_txt_feature', False),
                'only_resize': default_getattr(args, 'datasets_cfg.med_harvard.only_resize', None),
                'device': 'cpu'  # pin memory in dataloader
            }
            
            args.fusion_task = 'medical_fusion'  # 'SPECT-MRI', 'PET-MRI', 'CT-MRI'
            train_ds = MedHarvardDataset(
                args.path.base_dir,
                mode="train",
                data_source="xmu",
                transform_ratio=args.aug_probs[0],
                task=task,
                **additional_args
            )
            val_ds = MedHarvardDataset(
                args.path.base_dir,
                mode="test",
                data_source="xmu",
                transform_ratio=args.aug_probs[1],
                task=task,
                **additional_args
            )
            
        elif args.dataset == "sice":
            from task_datasets import SICEDataset
            
            additional_args = {
                'only_y': default_getattr(args, 'datasets_cfg.sice.only_y', False),
                'use_gt': default_getattr(args, 'datasets_cfg.sice.use_gt', False),
                'with_txt': default_getattr(args, 'datasets_cfg.sice.with_txt', False),
                'with_mask': default_getattr(args, 'datasets_cfg.sice.with_mask', False),
                'stop_aug_when_n_iters': default_getattr(args, 'datasets_cfg.sice.stop_aug_when_n_iters', -1),
            }

            args.fusion_task = 'MEF'
            train_ds = SICEDataset(
                data_dir=args.path.base_dir,
                mode="all",
                transform_ratio=args.aug_probs[0],
                output_size=default_getattr(args, 'datasets_cfg.sice.output_size', 128),
                **additional_args
            )
            val_ds = SICEDataset(
                data_dir=args.path.base_dir,
                mode="test",
                transform_ratio=args.aug_probs[1],
                **additional_args
            )

        elif args.dataset == "mefb":
            from task_datasets import MEFBDataset
            additional_args = {
                'only_y': default_getattr(args, 'datasets_cfg.mefb.only_y', False),
                'with_mask': default_getattr(args, 'datasets_cfg.mefb.with_mask', False),
                'with_txt': default_getattr(args, 'datasets_cfg.mefb.with_txt', False),
                'stop_aug_when_n_iters': default_getattr(args, 'datasets_cfg.mefb.stop_aug_when_n_iters', -1),
            }
            
            args.fusion_task = 'MEF'
            train_ds = MEFBDataset(
                data_dir=args.path.base_dir,
                mode="all",
                transform_ratio=args.aug_probs[0],
                output_size=default_getattr(args, 'datasets_cfg.mefb.output_size', 128),
                **additional_args
            )
            val_ds = MEFBDataset(
                data_dir=args.path.base_dir,
                mode="test",
                transform_ratio=args.aug_probs[1],
                **additional_args
            )
        
        elif args.dataset == 'realmff':
            from task_datasets import RealMFFDataset
            additional_args = {
                'with_mask': default_getattr(args, 'datasets_cfg.realmff.with_mask', False),
                'with_txt': default_getattr(args, 'datasets_cfg.realmff.with_txt', False),
                'use_gt': default_getattr(args, 'datasets_cfg.realmff.use_gt', False),
                'stop_aug_when_n_iters': default_getattr(args, 'datasets_cfg.realmff.stop_aug_when_n_iters', -1),
            }
            
            args.fusion_task = 'MFF'
            train_ds = RealMFFDataset(
                data_dir=args.path.base_dir,
                mode="all",
                transform_ratio=args.aug_probs[0],
                output_size=default_getattr(args, 'datasets_cfg.realmff.output_size', 128),
                **additional_args
            )
            val_ds = RealMFFDataset(
                data_dir=args.path.base_dir,
                mode="test",
                transform_ratio=args.aug_probs[1],
                **additional_args
            )
        
        elif args.dataset == "mff_whu":
            from task_datasets import MFFWHUDataset
            additional_args = {
                'use_gt': default_getattr(args, 'datasets_cfg.mff_whu.use_gt', False),
                'with_mask': default_getattr(args, 'datasets_cfg.mff_whu.with_mask', False),
                'with_txt': default_getattr(args, 'datasets_cfg.mff_whu.with_txt', False),
            }
            
            args.fusion_task = 'MFF'
            train_ds = MFFWHUDataset(
                data_dir=args.path.base_dir,
                mode="all",
                transform_ratio=args.aug_probs[0],
                output_size=default_getattr(args, 'datasets_cfg.mff_whu.output_size', 128),
                **additional_args
            )
            val_ds = MFFWHUDataset(
                data_dir=args.path.base_dir,
                mode="test",
                transform_ratio=args.aug_probs[1],
                **additional_args
            )
            
        elif args.dataset == "unify_fusion":
            from task_datasets.unify_fusion_datasets import make_unify_dataloader
            args.fusion_task = None
            args.task = 'UnifyFusion'
            
            datasets_kwargs = {
                'root_of_dirs': default_getattr(args, 'datasets_cfg.unify_fusion_datasets.root_of_dirs', None),
                'aug_prob': default_getattr(args, 'datasets_cfg.unify_fusion_datasets.aug_prob', 0.3),
                # NOTE: hack `transforms`, `augmentations_pipes`, and `is_valid_file` mannualy in your main script
                'transforms': default_getattr(args, 'datasets_cfg.unify_fusion_datasets.transforms', None),
                'augmentations_pipes': default_getattr(args, 'datasets_cfg.unify_fusion_datasets.augmentations_pipes', None),
                'is_valid_file': default_getattr(args, 'datasets_cfg.unify_fusion_datasets.is_valid_file', None),
            }
            dataloader_kwargs = {
                'batch_size': args.train_bs,
                'num_workers': args.num_workers,
                'pin_memory': True,
                'shuffle': args.shuffle if not state.use_distributed else None,
                'drop_last': True if args.shuffle else False,
            }
            train_dl = make_unify_dataloader(
                root_of_dirs=args.path.base_dir,
                datasets_kwargs=datasets_kwargs,
                dataloader_kwargs=dataloader_kwargs,
            )
            val_ds = None
            args._construct_dataloader = False
        else:
            raise NotImplementedError(f"not support dataset {args.dataset}")

    elif args.dataset in [
        "wv3",
        "qb",
        "gf2",
        "cave_x4",
        "harvard_x4",
        "cave_x8",
        "harvard_x8",
        "hisi-houston",
        "pavia",
        "chikusei",
        "botswana",
    ]:
        args.task = "sharpening"

        # the dataset has already splitted
        # FIXME: 需要兼顾老代码（只有trian_path和val_path）的情况
        if hasattr(args.path, "train_path") and hasattr(args.path, "val_path"):
            # 旧代码：手动切换数据集路径
            train_path = args.path.train_path
            val_path = args.path.val_path
        else:
            _args_path_keys = list(args.path.__dict__.keys())
            for k in _args_path_keys:
                if args.dataset in k:
                    train_path = getattr(args.path, f"{args.dataset}_train_path")
                    val_path = getattr(args.path, f"{args.dataset}_val_path")
        assert train_path is not None and val_path is not None, \
                "train_path and val_path should not be None"
                
        h5_train, h5_val = (
            h5py.File(train_path),
            h5py.File(val_path),
        )
        
        if args.dataset in ["wv3", "qb"]:
            from task_datasets import WV3Datasets

            args.fusion_task = 'Pansharpening'
            
            txt_feature_online_load = True if args.dataset == "qb" else False
            d_train, d_val = h5py_to_dict(h5_train), h5py_to_dict(h5_val)
            train_ds, val_ds = (
                WV3Datasets(d_train,
                            aug_prob=args.aug_probs[0],
                            txt_file=default_none_getattr(args, f'path.{args.dataset}_txt_train_path'),
                            txt_feature_online_load=txt_feature_online_load),
                WV3Datasets(d_val,
                            aug_prob=args.aug_probs[1],
                            txt_file=default_none_getattr(args, f'path.{args.dataset}_txt_val_path'),
                            txt_feature_online_load=txt_feature_online_load)
            )
            
        elif args.dataset == "gf2":
            from task_datasets import GF2Datasets
            
            args.fusion_task = 'Pansharpening'
            d_train, d_val = h5py_to_dict(h5_train), h5py_to_dict(h5_val)
            train_ds, val_ds = (
                GF2Datasets(d_train, 
                            aug_prob=args.aug_probs[0],
                            txt_file=default_none_getattr(args, 'path.gf2_txt_train_path'),
                            txt_feature_online_load=True),
                GF2Datasets(d_val, 
                            aug_prob=args.aug_probs[1],
                            txt_file=default_none_getattr(args, 'path.gf2_txt_val_path'),
                            txt_feature_online_load=True),  
            )
            
        elif (args.dataset[:4] == "cave" or
              args.dataset[:7] == "harvard" or
              args.dataset[:8] == "chikusei" or
              args.dataset[:5] == "pavia" or
              args.dataset[:8] == "botswana" or
              args.dataset[:7] == "houston"):
            from task_datasets import HISRDatasets

            args.fusion_task = 'HMIF'
                
            if args.dataset.split("-")[-1] == "houston":
                from einops import rearrange
                
                def permute_fn(x):
                    return rearrange(x, "b h w c -> b c h w")
                dataset_fn = permute_fn
            else:
                dataset_fn = None
            
            ## get txt file path
            def get_dataset_txt_paths(mode, dataset_name):
                return {
                    # "cave":         f"path.cave_txt_{mode}_path",
                    # "harvard":      f"path.harvard_txt_{mode}_path",
                    "chikusei":     f"path.chikusei_txt_{mode}_path",
                    "pavia":        f"path.pavia_txt_{mode}_path",
                    "botswana":     f"path.botswana_txt_{mode}_path",
                    "houston":      f"path.houston_txt_{mode}_path",
                }.get(dataset_name, None)
            
            ## get dataset dict
            d_train, d_val = (
                h5py_to_dict(h5_train),
                h5py_to_dict(h5_val),
            )
            
            # large files, close it
            h5_train.close()
            h5_val.close()

            ## get txt path
            if args.dataset[:4] == "cave" or args.dataset[:7] == "harvard": 
                txt_file_train = None
                txt_file_val = None
            else:
                txt_file_path_train = get_dataset_txt_paths(mode="train", dataset_name=args.dataset)
                txt_file_path_val = get_dataset_txt_paths(mode="val", dataset_name=args.dataset)
                assert txt_file_path_train is not None and txt_file_path_val is not None, "txt_file_path_train and txt_file_path_val should not be None"
                
                txt_file_train = default_none_getattr(args, txt_file_path_train)
                txt_file_val = default_none_getattr(args, txt_file_path_val)
            
            ## get datasets
            train_ds = HISRDatasets(
                d_train, 
                aug_prob=args.aug_probs[0],
                txt_file=txt_file_train,
                dataset_fn=dataset_fn,
                dataset_name=args.dataset,
            )
            val_ds = HISRDatasets(
                d_val, 
                aug_prob=args.aug_probs[1],
                txt_file=txt_file_val,
                dataset_fn=dataset_fn,
                dataset_name=args.dataset,
            )
    elif args.dataset.endswith("vae"):  # 'vq_vae' or 'lfq_vae'
        from task_datasets.unify_fusion_datasets import make_unify_dataloader
        from torch.utils.data import random_split
        
        unify_dataset_kwargs = {
            'root_of_dirs': default_getattr(args, 'datasets_cfg.vae.root_of_dirs', None),
            'aug_prob': default_getattr(args, 'datasets_cfg.vae.aug_prob', 0.0),
        }
        unify_dataloader_kwargs = {
            'batch_size': args.train_bs,
            'num_workers': args.num_workers,
            'pin_memory': True,
            'shuffle': args.shuffle, #if not state.use_distributed else None,
            'drop_last': True, # if args.shuffle else False,
        }
        train_dl, _ = make_unify_dataloader(
            unify_dataset_kwargs=unify_dataset_kwargs,
            dataloader_kwargs=unify_dataloader_kwargs,
        )
        train_ds = train_dl.dataset
        # split as validation dataset but still in training dataloader
        val_ds = random_split(train_ds, [0.05, 0.95])[0]
        val_dl = DataLoader(val_ds, **unify_dataloader_kwargs)
        args._construct_dataloader = False
    
    elif args.dataset.startswith("dif"):  # 'dif_*' dataset
        from task_datasets.DIF.degraded_wds import get_degraded_wds_loader
        
        train_cfg = args.dataset_cfg.train
        val_cfg = args.dataset_cfg.val
        train_ds, train_dl = get_degraded_wds_loader(**train_cfg)
        val_ds, val_dl = get_degraded_wds_loader(**val_cfg)
        args._construct_dataloader = False
        
    else:
        raise NotImplementedError(f"not support dataset {args.dataset}")
    
    ## Dataloader
    if default_getattr(args, '_construct_dataloader', True):
        train_sampler, val_sampler = None, None
        seed = args.default_getattr('seed', None)
        loader_generator = torch.Generator().manual_seed(seed) if seed is not None else None
        n_worker = default_getattr(args, 'num_workers', 0)
        if train_dl is None and val_ds is not None:
            train_dl = DataLoader(
                train_ds,
                args.train_bs,
                num_workers=n_worker,
                sampler=train_sampler,
                pin_memory=True,
                shuffle=args.shuffle, #if not state.use_distributed else None,
                drop_last=True if args.shuffle else False,
                prefetch_factor=8 if n_worker > 0 else None,
                persistent_workers=True if n_worker > 0 else False,
                generator=loader_generator,
            )
        if val_dl is None and val_ds is not None:
            val_dl = DataLoader(
                val_ds,
                args.val_bs,  # assert bs is 1, when using PatchMergeModule
                num_workers=0,
                sampler=val_sampler,
                pin_memory=False,
                shuffle=args.shuffle, #if not state.use_distributed else None,
                drop_last=False,
                generator=loader_generator,
            )
    
    if '_construct_dataloader' in args:
        delattr(args, '_construct_dataloader')

    return train_ds, train_dl, val_ds, val_dl


from torch.utils.data import Dataset

class ChainDataset(Dataset):
    def __init__(self, datasets: list[Dataset]) -> None:
        super().__init__()
        self.datasets = datasets
        self.prod_len = np.cumsum([len(d) for d in datasets])
        self.prod_len = np.insert(self.prod_len, 0, 0)
        # self.sample_idx_to_dataset_idx = np.digitize(np.arange(self.prod_len[-1]), self.prod_len)
        self.sample_idx_to_dataset_idx = lambda x: np.searchsorted(self.prod_len, x, side='right') - 1
            
    def __getitem__(self, index):
        dataset_idx = self.sample_idx_to_dataset_idx(index)
        return self.datasets[dataset_idx][index - self.prod_len[dataset_idx]]
    
    def __len__(self):
        return self.prod_len[-1]
    

def concat_dataset(datasets: list[Dataset]):
    """concatenate datasets
    """
    assert len(datasets) > 0, "datasets should not be empty"
    
    return ChainDataset(datasets)


def get_train_dataset(main_args: "NameSpace | DictConfig",
                      init_with_default_ds_cfg: bool = True):
    """get train dataset
    
    args modified:
        _construct_dataloader: bool (delete after used)
        dataset: str (change if using multi-dataset)
    
    """
    from accelerate import PartialState
    
    # accelerate partial state
    state = PartialState()
    
    # initial dataset arguments
    if init_with_default_ds_cfg:
        args = omegaconf_create("configs/datasets/datasets.yaml")
        args = dict_to_namespace(OmegaConf.to_container(args))
        if isinstance(main_args, DictConfig):
            main_args = dict_to_namespace(OmegaConf.to_container(main_args))
        args = merge_args(args, main_args)
    else:
        args = main_args
        
    # add multi-dataset support
    if "+" in args.dataset:
        _ds_name = args.dataset
        dataset_names = args.dataset.split("+")
        train_datasets = []
        val_datasets = []
        args._construct_dataloader = False
        for dataset_name in dataset_names:
            assert dataset_name in ["sice", "mefb", "realmff", "mff_whu"], f"not support dataset {dataset_name}"
            args.dataset = dataset_name
            train_ds, _, val_ds, _ = get_fusion_dataset(args)
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
        train_ds = concat_dataset(train_datasets)
        val_ds = concat_dataset(val_datasets)
        
        # dataloader
        n_worker = default_getattr(args, 'num_workers', 0)
        train_sampler, val_sampler = None, None  # handled by accelerator when using ddp
        train_dl = DataLoader(
            train_ds,
            args.train_bs,
            num_workers=n_worker,
            sampler=train_sampler,
            pin_memory=True,   # TODO: if pin memory in dataset ready with cuda stream, it will raise error
            shuffle=True, #args.shuffle if state.use_distributed else None,
            drop_last=True if args.shuffle else False,
            prefetch_factor=8 if n_worker > 0 else None,
            persistent_workers=True if n_worker > 0 else False
        )
        val_dl = DataLoader(
            val_ds,
            args.val_bs,  # assert bs is 1, when using PatchMergeModule
            num_workers=0,
            sampler=val_sampler,
            pin_memory=False,
            shuffle=True, #args.shuffle if not state.use_distributed else None,
            drop_last=False,
        )
        args.dataset = _ds_name  # unchange dataset name for logging
                
        return (train_ds, train_dl, val_ds, val_dl), args
    else:
        # single dataset or tar-ed webdataset
        return get_fusion_dataset(args), args


def set_ema_model_params_with_keys(ema_model_params: "dict[str, list[torch.Tensor] | int | float]", 
                                   keys: "list[str]",
                                   keys_set: list[str]=['shadow_params']):
    """set ema model parameters with keys

    Args:
        ema_model_params (dict[str, list[torch.Tensor] | int | float]): ema model parameters
        keys (list[str]): keys

    Returns:
        dict: ema model parameters with keys
    """
    logger = easy_logger()
    
    if not isinstance(keys, list):
        keys = list(keys)
    
    ema_model_params_with_keys = OrderedDict()
    for k in ema_model_params.keys():
        if k in keys_set and k in ema_model_params:
            logger.info(f'set ema_model {k} params with keys')
            params = ema_model_params[k]
            assert params is not None
            assert len(params) == len(keys), "ema_model_params and keys should have the same length"
            
            _params = OrderedDict()
            for mk, p in zip(keys, params):
                _params[mk] = p
                
            ema_model_params_with_keys[k] = _params
        elif k not in keys_set and k in ema_model_params:
            ema_model_params_with_keys[k] = ema_model_params[k]
            
    return ema_model_params_with_keys


def run_once(abled=True):
    def _inner(func):
        def _wrapper(*args, **kwargs):
            nonlocal abled
            if not abled:
                return None
            else:
                outs = func(*args, **kwargs)
                abled = False
                return outs

        return _wrapper

    return _inner


def sanity_check(func: callable):
    @run_once()
    def _inner(*args, **kwargs):
        return func(*args, **kwargs)

    return _inner


@contextmanager
def save_imgs_in_zip(
    zipfile_name: str, mode="w", verbose: bool = False, save_file_ext: str = "jpeg"
):
    """save images to a zip file

    Args:
        zipfile_name (str): zip filename
        mode (str, optional): mode to write in. Defaults to "w".
        verbose (bool, optional): print out. Defaults to False.
        save_file_ext (str, optional): image extension in the zip file. Defaults to "jpeg".

    Yields:
        callable: a function to save image

    Examples::

        with save_imgs_in_zip('zip_file.zip') as add_image:
            img, img_name = get_img()
            add_image(img, img_name)

    :ref: `add_image`

    """
    logger = easy_logger()
    
    # save_file_ext = save_file_ext.upper()
    zf = zipfile.ZipFile(
        zipfile_name, mode=mode, compression=zipfile.ZIP_DEFLATED, compresslevel=9
    )
    bytes_io = BytesIO()
    # jpg compression
    _jpg_quality = 100  # 95 if save_file_ext in ["jpeg", "jpg", "JPG", "JPEG"] else 100

    try:

        logger.info(f"zip file will be saved at {zipfile_name}")

        def to_bytes(image_data, image_name):
            batched_image_bytes = []

            if image_data.ndim == 4:  # batched rgb images
                assert isinstance(image_name, list), "image_name should be a list"
                assert image_data.shape[0] == len(
                    image_name
                ), "image_name should have the same length as image_data"

                for img in image_data:  # [b, h, w, c]
                    Image.fromarray(img).save(
                        bytes_io, format=save_file_ext, quality=_jpg_quality
                    )
                    batched_image_bytes.append(bytes_io.getvalue())
            elif image_data.ndim == 3:
                if image_data.shape[-1] == 1:  # gray image  # [h, w, 1]
                    Image.fromarray(image_data[..., 0]).save(
                        bytes_io, format=save_file_ext, quality=_jpg_quality
                    )
                    image_data = bytes_io.getvalue()
                elif image_data.shape[-1] == 3:
                    Image.fromarray(image_data).save(
                        bytes_io, format=save_file_ext, quality=_jpg_quality
                    )
                    image_data = bytes_io.getvalue()
                else:
                    raise ValueError(
                        f"image_data shape {image_data.shape} not supported"
                    )
            elif image_data.ndim == 2:  # gray image  # [h, w]
                Image.fromarray(image_data).save(
                    bytes_io, format=save_file_ext, quality=_jpg_quality
                )
                image_data = bytes_io.getvalue()

            return image_data, batched_image_bytes

        def add_image(
            image_data: "Image.Image | np.ndarray | torch.Tensor | bytes",
            image_name: "Union[str, list[str]]",
        ):
            """add image to the zipfile

            Args:
                image_data (Image.Image | np.ndarray | torch.Tensor | bytes): can be Image.Image, np.ndarray, torch.Tensor, bytes,
                                                    shape should be [b, h, w, c], [h, w, c], [h, w, 1]
                image_name (str | list[str]): saved image names
            """

            # to bytes
            batched_image_bytes = None
            if isinstance(image_data, Image.Image):
                image_data.save(bytes_io, format=save_file_ext)
                bytes = bytes_io.getvalue()
            elif isinstance(image_data, np.ndarray):
                bytes, batched_image_bytes = to_bytes(image_data, image_name)
            elif isinstance(image_data, torch.Tensor):
                image_data = image_data.detach().cpu().numpy()
                bytes, batched_image_bytes = to_bytes(image_data, image_name)
            else:
                raise ValueError(f"image_data type {type(image_data)} not supported")

            # saving to zip file
            if batched_image_bytes is not None:
                for i, img_bytes in enumerate(batched_image_bytes):
                    zf.writestr(image_name[i], img_bytes)
            else:
                zf.writestr(image_name, bytes)

            if verbose:
                logger.info(f"add image {image_name} to zip file")

            bytes_io.seek(0)
            bytes_io.truncate()

        yield add_image

    except Exception as e:
        if verbose:
            logger.error(e, raise_error=True)
            raise e
    finally:
        if verbose:
            logger.info(f"zip file saved at {zipfile_name}, zipfile close")
        zf.close()
        bytes_io.close()

if __name__ == "__main__":
    # from pathlib import Path
    # from PIL import Image

    # from utils import EasyProgress

    # vi_path = Path("/Data3/cao/ZiHanCao/datasets/M3FD/M3FD_Fusion/raw_png/ir")
    # img_paths = list(vi_path.glob("*.png"))
    # tbar, task = EasyProgress.easy_progress(["save images in zip"], [len(img_paths)])

    # tbar.start()
    # with save_imgs_in_zip("ir_jpg.zip", verbose=False) as add_img:
    #     for p in img_paths:
    #         img = Image.open(p)
    #         saved_name = p.name.replace(".png", ".jpg")
    #         add_img(img, saved_name)
    #         tbar.update(
    #             task, advance=1, total=len(img_paths), description=f"saved {saved_name}"
    #         )

    
    # from omegaconf import OmegaConf
    
    # conf = OmegaConf.load("configs/datasets/datasets.yaml")
    # print(conf)
    
        
    import accelerate
    accelerator = accelerate.Accelerator()
    
    with catch_any_error():
        conf = omegaconf_create("configs/panRWKV_config.yaml")
        conf.dataset = 'sice+mefb'
        conf.train_bs = 8
        conf.val_bs = 1
        conf.shuffle = False
        conf.ddp = False
        conf.aug_probs = [0.0, 0.0]
        conf.fusion_crop_size = 256
        
        ds_dl, args = get_train_dataset(conf)
        train_dl = ds_dl[1]
        val_dl = ds_dl[-1]
        
        train_dl, val_dl = accelerator.prepare(train_dl, val_dl)
        
        for i, data in enumerate(train_dl, 1):
            print(f'[{i}/{len(train_dl)}]', data.keys(), data['gt'].shape)
        
        print('-' * 100)
        for i, data in enumerate(val_dl, 1):
            print(f'[{i}/{len(val_dl)}]', data.keys(), data['gt'].shape)