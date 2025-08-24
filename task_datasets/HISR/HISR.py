from typing import Union
import torch
import torch.utils.data as data
import torchvision.transforms as T
import h5py
from typing import List, Tuple, Optional, Callable
from utils import easy_logger
from safetensors.numpy import load_file  # Added for loading txt features
from safetensors import safe_open

logger = easy_logger(func_name='HISR')

def default_dataset_fn(*x, dataset_name=None):
    """
    默认数据集预处理函数，包含clip和归一化处理
    """
    data = x[0]
    if isinstance(data, torch.Tensor):
        # 1. 首先clip负值 - 对于光谱数据，负值通常是噪声或异常值
        data = torch.clamp(data, min=0.0)

        data_max = data.max()
        data_min = data.min()  # 由于已经clip，这里min应该是0或正数
        
        # 特殊处理Harvard多尺度数据集
        harvard_mulit_datasets = [
            'harvard_mulit_x4',
            'harvard_mulit_x8', 
            'harvard_mulit_x16',
            'harvard_mulit_x32'
        ]
        
        # 判断是否需要强制归一化
        force_normalize = dataset_name in harvard_mulit_datasets
        
        if data_max > 1.0 or force_normalize:
            # 如果最大值大于1，或者是指定的Harvard数据集，进行归一化
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
            else:
                # 处理常数张量的情况
                data = torch.zeros_like(data)
        # 如果数据已经在[0,1]范围内且不是Harvard数据集，保持不变
    
    # 3. 确保数据类型为float32
    if data.dtype != torch.float32:
        data = data.float()
    
    # 4. 最终clip确保数据在[0,1]范围内（防止数值计算误差）
    data = torch.clamp(data, min=0.0, max=1.0)
    
    return data
    ###原来的数据集
    # return x[0]

DATASET_KEY_MAPPING = {
    'cave': ['GT', 'LRHSI', 'RGB', 'HSI_up'],
    'harvard': ['GT', 'LRHSI', 'RGB', 'HSI_up'],
    'cave_x4': ['GT', 'LRHSI', 'RGB', 'HSI_up'],
    'cave_x8': ['GT', 'LRHSI', 'RGB', 'HSI_up'],
    'harvard_x4': ['GT', 'LRHSI', 'RGB', 'HSI_up'],
    'harvard_x8': ['GT', 'LRHSI', 'RGB', 'HSI_up'],
    'chikusei': ['GT', 'LRHSI', 'RGB', 'HSI_up'],
    'houston': ['GT', 'LRHSI', 'RGB', 'HSI_up'],
    'pavia': ['GT', 'MS', 'PAN', 'LMS'],
    'botswana': ['GT', 'MS', 'PAN', 'LMS'],
    'cave_mulit_x4': ['GT', 'LRHSI_4', 'HRMSI', 'lms_4'],
    'cave_mulit_x8': ['GT', 'LRHSI_8', 'HRMSI', 'lms_8'],
    'cave_mulit_x16': ['GT', 'LRHSI_16', 'HRMSI', 'lms_16'],
    'cave_mulit_x32': ['GT', 'LRHSI_32', 'HRMSI', 'lms_32'],
    'harvard_mulit_x4': ['GT', 'LRHSI_4', 'HRMSI', 'lms_4'],
    'harvard_mulit_x8': ['GT', 'LRHSI_8', 'HRMSI', 'lms_8'],
    'harvard_mulit_x16': ['GT', 'LRHSI_16', 'HRMSI', 'lms_16'],
    'harvard_mulit_x32': ['GT', 'LRHSI_32', 'HRMSI', 'lms_32'],
    'chikusei_mulit_x4': ['GT', 'LRHSI_4', 'HRMSI', 'lms_4'],
    'chikusei_mulit_x8': ['GT', 'LRHSI_8', 'HRMSI', 'lms_8'],
    'chikusei_mulit_x16': ['GT', 'LRHSI_16', 'HRMSI', 'lms_16'],
    'chikusei_mulit_x32': ['GT', 'LRHSI_32', 'HRMSI', 'lms_32'],
    'paviac_mulit_x4': ['GT', 'LRHSI_4', 'HRMSI', 'lms_4'],
    'paviac_mulit_x8': ['GT', 'LRHSI_8', 'HRMSI', 'lms_8'],
    'paviac_mulit_x16': ['GT', 'LRHSI_16', 'HRMSI', 'lms_16'],
    'paviac_mulit_x32': ['GT', 'LRHSI_32', 'HRMSI', 'lms_32'],
    'paviau_mulit_x4': ['GT', 'LRHSI_4', 'HRMSI', 'lms_4'],
    'paviau_mulit_x8': ['GT', 'LRHSI_8', 'HRMSI', 'lms_8'],
    'paviau_mulit_x16': ['GT', 'LRHSI_16', 'HRMSI', 'lms_16'],
    'paviau_mulit_x32': ['GT', 'LRHSI_32', 'HRMSI', 'lms_32'],
    'botswana_mulit_x4': ['GT', 'LRHSI_4', 'HRMSI', 'lms_4'],
    'botswana_mulit_x8': ['GT', 'LRHSI_8', 'HRMSI', 'lms_8'],
    'botswana_mulit_x16': ['GT', 'LRHSI_16', 'HRMSI', 'lms_16'],
    'botswana_mulit_x32': ['GT', 'LRHSI_32', 'HRMSI', 'lms_32'],
}

class HISRDatasets(data.Dataset):
    """
    Hyspectral and multispectral image datasets for image fusion
    """
    
    def __init__(
        self,
        file: Union[h5py.File, str, dict],
        aug_prob=0.0,
        rgb_to_bgr=False,
        full_res=False,
        txt_file=None,
        *,
        dataset_fn=None,
        dataset_name: str | None=None,
    ):
        super(HISRDatasets, self).__init__()
        # warning: you should not save file (h5py.File) in this class,
        # or it will raise CAN NOT BE PICKLED error in multiprocessing
        # FIXME: should pass @path rather than @file which is h5py.File object to avoid can not be pickled error
        if isinstance(file, (str, h5py.File)):
            if isinstance(file, str):
                file = h5py.File(file)
            logger.warning(
                "when @file is a h5py.File object, it can not be pickled. ",
                "try to set DataLoader number_worker to 0"
            )
        self.dataset_name = dataset_name
        assert dataset_name is not None, f"dataset_name should be provided, choices can be {list(DATASET_KEY_MAPPING.keys())}"
        
        # checking dataset_fn type
        if dataset_fn is not None:
            if isinstance(dataset_fn, (list, tuple)):
                def _apply_fn(tensor):
                    for fn in dataset_fn:
                        tensor = fn(tensor)
                    return tensor
                self.dataset_fn = _apply_fn 
            elif callable(dataset_fn):
                self.dataset_fn = dataset_fn
            else: 
                raise TypeError("dataset_fn should be a list of callable or a callable object")
        else:
            self.dataset_fn = default_dataset_fn
                    
        self.full_res = full_res
        data_s= self._split_parts(
            file, rgb_to_bgr=rgb_to_bgr, full=full_res
        )
        
        if len(data_s) == 4:
            self.gt, self.lr_hsi, self.rgb, self.hsi_up = data_s
        else:
            self.lr_hsi, self.rgb, self.hsi_up = data_s           
        
        self.size = self.rgb.shape[-2:]
        logger.info("dataset shape:")

        # log dataset info
        if not full_res:
            logger.info("{:^20}{:^20}{:^20}{:^20}".format("lr_hsi", "hsi_up", "rgb", "gt"))
            logger.info(
                "{:^20}{:^20}{:^20}{:^20}".format(
                    str(tuple(self.lr_hsi.shape)),
                    str(tuple(self.hsi_up.shape)),
                    str(tuple(self.rgb.shape)),
                    str(tuple(self.gt.shape)),
                )
            )
        else:
            logger.info("{:^20}{:^20}{:^20}".format("lr_hsi", "hsi_up", "rgb"))
            logger.info(
                "{:^20}{:^20}{:^20}".format(
                    str(tuple(self.lr_hsi.shape)),
                    str(tuple(self.hsi_up.shape)),
                    str(tuple(self.rgb.shape)),
                )
            )
        
        # load txt features if provided
        self.with_txt = True if txt_file is not None else False
        if self.with_txt:
            # self.txt_features = load_file(txt_file)
            self.txt_features = safe_open(txt_file, framework='numpy')
            assert len(self.txt_features.keys()) == len(self.rgb), ('size of txt features and rgb images should be the same, '
                                                             'but got {} and {}'.format(len(self.txt_features.keys()), len(self.rgb)))

        # geometrical transformation
        self.aug_prob = aug_prob
        if aug_prob != 0.0:
            self.geo_trans = (T.Compose([
                T.RandomApply([
                                T.RandomErasing(
                                    p=self.aug_prob,
                                    scale=(0.02, 0.15),
                                    ratio=(0.2, 1.0)
                                ),
                                T.RandomAffine(
                                    degrees=(0, 70),
                                    translate=(0.1, 0.2),
                                    scale=(0.95, 1.2),
                                    interpolation=T.InterpolationMode.BILINEAR,
                                )], 
                            p=self.aug_prob),
                ]) 
            )

    def _split_parts(self, file, load_all=True, rgb_to_bgr=False, keys=None, full=False):
        
        # warning: key RGB is HRMSI when the dataset is GF5-GF1
        if not full:
            keys = DATASET_KEY_MAPPING[self.dataset_name]
        else:
            keys = DATASET_KEY_MAPPING[self.dataset_name].pop(0)
        logger.info(f"load dataset keys: {keys}")
        
        if load_all:
            # load all data in memory
            data = []
            # breakpoint()
            for k in keys:
                # logger.info(f'load key {k}')
                data.append(
                    self.dataset_fn(torch.as_tensor(file[k][:], dtype=torch.float32), dataset_name=self.dataset_name),
                )
                
            if rgb_to_bgr:
                logger.warning("rgb to bgr, for testing generalization only.")
                # rgb -> bgr
                if not full:
                    data[2] = data[2][:, [-1, 1, 0]]
                else:
                    data[1] = data[1][:, [-1, 1, 0]]
            return data
        else:
            return [file.get(k) for k in keys]

    def aug_trans(self, data: dict):
        # we set a seed from the training start
        _random_state = torch.random.get_rng_state()
        data_dict = {}
        for k, d in data.items():
            torch.random.set_rng_state(_random_state)
            d = self.geo_trans(d)
            data_dict[k] = d
        return data_dict

    def __getitem__(self, index):
        # gt: [31, 64, 64]
        # lr_hsi: [31, 16, 16]
        # rbg: [3, 64, 64]
        # hsi_up: [31, 64, 64]

        # harvard [rgb]
        # cave [bgr]
        
        outp = {
            'rgb': self.rgb[index],
            'lr_hsi': self.lr_hsi[index],
            'hsi_up': self.hsi_up[index],
        }
        if not self.full_res:
            outp['gt'] = self.gt[index]
        
        if self.with_txt:
            # outp['txt'] = torch.from_numpy(
            #     self.txt_features[str(index)][0]
            # ).type(torch.float32)
            outp['txt'] = torch.from_numpy(
                self.txt_features.get_tensor(str(index))[0]
            ).type(torch.float32)
        
        if self.aug_prob != 0.0 and not self.with_txt:
            outp = self.aug_trans(outp)
        
        return outp

    def __len__(self):
        return len(self.rgb)


if __name__ == "__main__":
    path = r"/data2/users/yujieliang/exps/Efficient-MIF-back-master-6-feat/data/ASSR/Harvard/Harvard_test_crop1024.h5"
    file = h5py.File(path)
    dataset = HISRDatasets(file, aug_prob=0., txt_file=None,
                           dataset_name='harvard_mulit_x4')  # Specify the txt file path
    dl = data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True
    )
    # from tqdm import tqdm
    # for i, data in tqdm(enumerate(dl, 1)):

    #     logger.info(
    #         f"lr_hsi: {lr_hsi.shape}, rgb: {rgb.shape}, hsi_up: {hsi_up.shape}, gt: {gt.shape}",
    #     )
    #     fig, axes = plt.subplots(ncols=4, figsize=(20, 5))
    #     axes[0].imshow(rgb[0].permute(1, 2, 0).numpy()[..., :3])
    #     axes[1].imshow(lr_hsi[0].permute(1, 2, 0).numpy()[..., :3])
    #     axes[2].imshow(hsi_up[0].permute(1, 2, 0).numpy()[..., :3])
    #     axes[3].imshow(gt[0].permute(1, 2, 0).numpy()[..., :3])
    #     plt.tight_layout(pad=0)
    #     plt.show()
    #     time.sleep(3)
    #     fig.savefig(f'./tmp/{i}.png', dpi=100)
    #     pass