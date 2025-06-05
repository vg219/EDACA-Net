import webdataset as wds
import numpy as np
import torch as th
import json
from tqdm import trange

from task_datasets.DIF.make_degraded_instruction_pairs import numpy_img_to_bytes


def make_fake_data(
    tar_file: str,
    n_samples: int=200,
) -> None:
    # llm instruction
    instruction = json.dumps({"instruction": "A beautiful landscape painting."})
    encoded_instruction_fn = lambda: np.random.randn(512, 77) # clip size
    
    # degraded images
    degraded_img_fn = lambda: numpy_img_to_bytes(np.random.rand(256, 256, 3))
    gt_img_fn = lambda: numpy_img_to_bytes(np.random.rand(256, 256, 3))
    encoded_degraded_img_fn = lambda: np.random.randint(-1, 1, (16, 16, 18)).astype(np.int32)
    encoded_gt_img_fn = lambda: np.random.randint(-1, 1, (16, 16, 18)).astype(np.int32)
    
    with wds.TarWriter(tar_file) as tar:
        for i in trange(n_samples):
            tar.write(
                {
                    "__key__": str(i),
                    "instruction.json": instruction, 
                    "instruction.npy": encoded_instruction_fn(), 
                    "encoded_m1.npy": encoded_degraded_img_fn(), 
                    "encoded_m2.npy": encoded_gt_img_fn(), 
                    "m1.jpg": degraded_img_fn(), 
                    "m2.jpg": gt_img_fn()
                }
            )
            
if __name__ == "__main__":
    make_fake_data(tar_file="/Data3/cao/ZiHanCao/exps/panformer/task_datasets/DIF/test_tar_file.tar")
    