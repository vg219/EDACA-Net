__all__ = [
    'WV3Datasets',
    'GF2Datasets',
    'HISRDatasets',
    'LLVIPDatasets',
    'LLVIPDALIPipeLoader',
    'LytroDataset',
    'M3FDDALIPipeLoader',
    'MedHarvardDataset',
    'MSRSDatasets',
    'MSRSDALIPipeLoader',
    'RoadSceneDataset',
    'SICEDataset',
    'TNODataset',
    'VISIRJointGenericLoader',
    'RealMFFDataset',
    'MEFBDataset',
    'MFFWHUDataset',
    'LytroDataset',
]

# datasets cls
## pansharpening
from .Pansharpening.WV3 import WV3Datasets
from .Pansharpening.GF2 import GF2Datasets
## HMIF
from .HISR.HISR import HISRDatasets
## VIF
from .VIF.TNO import TNODataset
from .VIF.RoadScene import RoadSceneDataset
from .VIF.MSRS import MSRSDALIPipeLoader, MSRSDatasets
from .VIF.M3FD import M3FDDALIPipeLoader
from .VIF.LLVIP import LLVIPDALIPipeLoader, LLVIPDatasets
from .VIF.VIS_IR_joint_pipe import VISIRJointGenericLoader
## Medical Image Fusion
from .MIF.MedHarvard import MedHarvardDataset
## MEF
from .MEF.MEFB import MEFBDataset
from .MEF.SICE import SICEDataset
from .MFF.Lytro import LytroDataset
## MFF
from .MFF.Real_MFF import RealMFFDataset
from .MFF.MFF_WHU import MFFWHUDataset
# simple paired dataset
from .paired_dataset import PairedDataset


# dataset output keys with fusion task specified
DATASET_KEYS = {
    'pansharpening': ['ms', 'lms', 'pan', 'gt', 'txt'],
    'HMIF': ['rgb', 'lr_hsi', 'hsi_up', 'gt', 'txt'],
    'VIF': ['vi', 'ir', 'mask', 'gt', 'txt', 'name'],
    'MEF': ['over', 'under', 'gt', 'txt', 'name'],
    'MFF': ['far', 'near', 'gt', 'txt', 'name'],
    'medical_fusion': ['s1', 's2', 'mask', 'gt', 'txt', 'name'],
}
