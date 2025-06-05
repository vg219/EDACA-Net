"""

    # with wds.ShardWriter("/Data3/cao/ZiHanCao/datasets/VIF-LLVIP/webdataset/LLVIP_train_%06d.tar", maxcount=1000) as f:
    #     infrared = os.path.join(path, 'infrared')
    #     visible = os.path.join(path, 'visible')
    #     
    #     for i, file in enumerate(file_names):
    #         infrared_path = os.path.join(infrared, 'train', file)
    #         visible_path = os.path.join(visible, 'train', file)
    #         
    #         logger.info(f'Processing [{i}/{len(file_names)}] {os.path.basename(infrared_path)}')
    #         
    #         # load image and convert to bytes using BytesIO
    #         with open(visible_path, 'rb') as img_f:
    #             vis_io_to_wds = img_f.read()
    #         with open(infrared_path, 'rb') as img_f:
    #             ir_io_to_wds = img_f.read()
    #         
    #         # to wds
    #         wds_data = {
    #             '__key__': os.path.basename(infrared_path).split('.')[0],
    #             'vis.jpg': vis_io_to_wds,
    #             'ir.jpg': ir_io_to_wds,
    #             'json': {
    #                 'name': os.path.basename(infrared_path),
    #             }
    #         }
    #         
    #         f.write(wds_data)
"""



from accelerate import Accelerator
from accelerate.utils import set_seed
import PIL.Image as Image
import numpy as np
import torch
import torch.distributed
from torchvision import transforms
import webdataset as wds
import os
from io import BytesIO
from loguru import logger
import torch.multiprocessing as mp

def main():
    # 初始化 accelerator
    accelerator = Accelerator()
    
    # 设置随机种子以确保可重复性
    set_seed(42)
    
    logger.info(f"Process details: {accelerator.process_index}/{accelerator.num_processes}")
    
    path = '/Data3/cao/ZiHanCao/datasets/VIF-LLVIP/data'
    file_names = os.listdir(os.path.join(path, 'infrared', 'train'))

    # read to test the wds file
    wds_file = '/Data3/cao/ZiHanCao/datasets/VIF-LLVIP/webdataset/LLVIP_train_000000.tar'
    
    in_code_transform = transforms.Compose([
        transforms.RandomCrop(size=(256, 256)),
        transforms.ToTensor()
    ])

    wds_dataset = (
        wds.WebDataset(wds_file, shardshuffle=False, resampled=True, verbose=True, empty_check=False, nodesplitter=wds.split_by_node)
        .decode('pil')
        .to_tuple('vis.jpg', 'ir.jpg', 'json')
        .map_tuple(in_code_transform, in_code_transform, lambda x: x)
        # .with_epoch(100)
    )
    
    data_loader = wds.WebLoader(
        wds_dataset, 
        num_workers=0, 
        batch_size=10, 
        # persistent_workers=True, 
        # prefetch_factor=2, 
        drop_last=False
    )
    
    # 使用 accelerator 准备数据加载器
    # data_loader = accelerator.prepare(data_loader)

    from tqdm import tqdm
    i = 0
    for vis, ir, json_data in tqdm(data_loader, total=30):
        i += 1
        if i % 10 == 0:
            logger.info(f'process {accelerator.process_index} has processed {i} images already')
            logger.info(f'vis.shape: {vis.shape}, ir.shape: {ir.shape}')

    logger.info(f'process {accelerator.process_index} has processed {i} images in total')

if __name__ == "__main__":
    main()



