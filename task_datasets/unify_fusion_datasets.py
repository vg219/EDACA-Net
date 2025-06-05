"""
Author: Zihan Cao
Date: 2024-10-10

Copyright (c) 2024 by Zihan Cao, All Rights Reserved.

"""


"""
load images for MEF, MFF, VIF, and other fusion tasks directory by directory

suitable for training or finetuning VQ-VAE models or other generative models 
with unsupervised loss

Have fun.

"""


import torch
import numpy as np
import os
import os.path as osp

from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torchvision.transforms.functional import InterpolationMode
from typing import Callable
from kornia.augmentation import (
    RandomResizedCrop,
    RandomCutMixV2,
    RandomHorizontalFlip, 
    RandomVerticalFlip,
    ColorJiggle,
    RandomGaussianNoise,
    RandomAffine,
    RandomGrayscale,
    RandomSnow,
    RandomRain,
    RandomErasing,
    AugmentationSequential,
)
from kornia.augmentation.auto import AutoAugment
from kornia.constants import Resample

import sys
sys.path.append('./')
from utils import easy_logger, default

logger = easy_logger(func_name='unify_fusion_datasets')

FUSION_DATA_ROOTS = [
    'VIF-M3FD',
    'VIF-LLVIP',
    'VIF-MSRS',
    'VIF-TNO',
    'VIF-RoadSceneFusion',
    'MEF-MEFB',
    'MEF-SICE',
    'MFF-Lytro',
    'MFF-RealMFF',
    'MFF-WHU',
]

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def make_collate_fn(augmentations_pipes: Callable):
    ## dataset collate function
    def collate_fn(batch):
        # move to device
        imgs, labels = [], []
        for img, label in batch:
            # img = img.cuda(non_blocking=True)
            imgs.append(img)
            labels.append(label)
        imgs = torch.stack(imgs, dim=0)
        imgs = augmentations_pipes(imgs)
        
        return imgs, labels
    
    return collate_fn

def _is_valid_file(x: str):
    return (
        'Segmentation' not in x and
        'raw_png' not in x and
        'labels' not in x and
        'Annotations' not in x and
        'mask' not in x and
        'txt' not in x and
        'mean_iou' not in x and
        'old' not in x and
        # ('MEF' in x or
        # 'MFF' in x or
        # 'VIF' in x or
        # 'Med' in x ) and
        x.endswith(IMG_EXTENSIONS)
    )
    

def make_unify_datasets(
    root_of_dirs: str,
    data_roots: list[str]=FUSION_DATA_ROOTS,
    aug_prob: float=0.3,
    transforms: "Callable | None"=None,
    augmentations_pipes: "Callable | str"="auto",
    is_valid_file: "Callable | None"=None,
):
    """
    make unify datasets for fusion tasks
    
    Args:
        root_of_dirs: str, the root of the directories
        data_roots: list[str], the data roots (names of the subdirectories)
        aug_prob: float, the probability of augmentation
        transforms: Callable | None, the transforms
        augmentations_pipes: Callable | None, the augmentations pipes
        is_valid_file: Callable | None, the is valid file
    """

    if transforms is None:
        transforms = Compose([
            ToTensor(),
            RandomResizedCrop(size=(256, 256), scale=(0.7, 1.),
                              ratio=(0.5, 2), keepdim=True, resample=Resample.BILINEAR),
        ])
        
    if augmentations_pipes == 'default':
        augmentations_pipes = AugmentationSequential(
            RandomSnow(p=aug_prob, snow_coefficient=(0.1, 0.3), brightness=(1, 3)),
            RandomRain(p=aug_prob, drop_height=(2, 5), drop_width=(-3, 3)),
            RandomGaussianNoise(std=0.1, p=aug_prob),
            RandomAffine(p=aug_prob, degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5), padding_mode='reflection'),
            RandomGrayscale(p=aug_prob, keepdim=True),
            RandomCutMixV2(p=aug_prob, num_mix=1, cut_size=(0.2, 0.6)),
            ColorJiggle(0.1, 0.1, 0.1, 0.1, p=aug_prob),
            RandomErasing(p=aug_prob, scale=(0.02, 0.1), ratio=(0.3, 2.0), value=0.5),
            keepdim=True,
            random_apply=(1, 3),
            same_on_batch=False,
            random_apply_weights=(0.2, 0.2, 1, 1, 0.8, 1, 0.5),
        )            
    elif augmentations_pipes == 'auto':
        augmentations_pipes = AugmentationSequential(AutoAugment(), same_on_batch=False)
    else:
        assert callable(augmentations_pipes), 'augmentations_pipes must be callable'
    
    is_valid_file = default(is_valid_file, _is_valid_file)
    assert callable(is_valid_file), 'is_valid_file must be callable'
    
    ## image folder
    def make_folders(roots: list[str],
                     is_valid_file: "Callable | None"=None,
                     transform: "Callable | None"=None,
                     allow_empty: bool=True):
        img_folders = [
            ImageFolder(root,
                        is_valid_file=is_valid_file,
                        allow_empty=allow_empty,
                        transform=transform) for root in roots
        ]
        
        return ConcatDataset(img_folders)
    
    logger.info(f'finding image in folder: {root_of_dirs} ...')
    dataset = make_folders([osp.join(root_of_dirs, da) for da in data_roots], is_valid_file, transforms)
    logger.info(f'found {len(dataset)} images')
    
    ## collate function
    collate_fn = make_collate_fn(augmentations_pipes)
    
    return dataset, collate_fn, augmentations_pipes

def make_unify_dataloader(
    unify_dataset_kwargs: dict,
    dataloader_kwargs: dict,
    *,
    augmentation_in_collate_fn: bool=False,
):
    """
    make a unify dataloader on fusion images
    
    Args:
        unify_dataset_kwargs: dict, the kwargs for make_unify_datasets
        dataloader_kwargs: dict, the kwargs for DataLoader
    """
    dataset, collate_fn, augmentations_pipes = make_unify_datasets(**unify_dataset_kwargs)
    
    if 'collate_fn' not in dataloader_kwargs and augmentation_in_collate_fn:
        dataloader_kwargs['collate_fn'] = collate_fn
        main_process_aug = lambda img: img
    else:
        main_process_aug = augmentations_pipes
        
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    
    # force aug pipes in dataloader
    dataloader._augmentations_pipes = main_process_aug
    
    return dataloader, main_process_aug


if __name__ == '__main__':
    root_of_dirs = '/Data3/cao/ZiHanCao/datasets'
    # dataset, collate_fn, augmentations_pipes = make_unify_datasets(root_of_dirs)
    # dataloader = DataLoader(dataset, batch_size=32, num_workers=2, shuffle=True, collate_fn=collate_fn)
    ds_kwargs = dict(
        root_of_dirs=root_of_dirs,
        aug_prob=0.3,
        transforms=None,
        augmentations_pipes='auto',
    )
    
    # dataloader kwargs
    dl_kwargs = dict(
        batch_size=1,
        num_workers=0,
        shuffle=True,
    )
    
    dataloader, _ = make_unify_dataloader(ds_kwargs, dl_kwargs, augmentation_in_collate_fn=True)
    main_process_aug = dataloader._augmentations_pipes
    
    import loguru
    
    with loguru.logger.catch():
        # import tqdm
        # for i, (img, label) in tqdm.tqdm(enumerate(dataloader)):
        #     print(img.shape, label)
        
        for img_aug, label in dataloader:
            img_aug = main_process_aug(img_aug)

            print(img_aug.shape, label)
            # break
            pass