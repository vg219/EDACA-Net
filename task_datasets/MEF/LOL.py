from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from kornia.augmentation import ColorJiggle, RandomResizedCrop, RandomGaussianBlur, RandomCrop, RandomBrightness, RandomContrast
from kornia.io import load_image, ImageLoadType
from rich.progress import track

import sys
sys.path.append('./')
from utils import easy_logger, is_main_process, default

logger = easy_logger()


# TODO: find how to split the train test data
class LOLDataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 transform: list[callable]=None,
                 transformer_ratio: float=0.,
                 output_size: int=128,
                 only_y: bool=False,
                 get_name: bool=False,
                 *,
                 mode: Literal['train', 'test']='train',
                 device: "th.device | str"=None):
        
        self.data_dir = Path(data_dir)
        self.only_y = only_y
        self.get_name = get_name
        self.mode = mode
        
        if only_y:
            logger.warning('output only Y channel of images')
            _load_type = ImageLoadType.GRAY8
        else:
            _load_type = ImageLoadType.RGB8
            
        load_keys = {'gt': 'GT', 'over': 'over', 'under': 'under'}