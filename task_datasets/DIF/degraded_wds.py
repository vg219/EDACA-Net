import numpy as np
import torch as th
from webdataset import WebDataset, WebLoader, shardlists
from accelerate.state import PartialState
from torchvision import transforms

from utils import easy_logger

logger = easy_logger(func_name='degraded_wds')

def pad_encoded_img(encoded_img: np.ndarray, pad_size: int=1024) -> np.ndarray:
    """Pad encoded image to the same size."""
    encoded_sz = encoded_img.shape
    H, W = encoded_sz[1:]
    
    assert H <= pad_size and W <= pad_size, f"Encoded image size {H}x{W} is larger than pad size {pad_size}x{pad_size}."
    
    pad_H = pad_size - H
    pad_W = pad_size - W
    
    padded_img = np.pad(encoded_img, ((0, 0), (0, pad_H), (0, pad_W)), mode='constant')
    
    return padded_img


def get_degraded_wds_loader(dataset_pattern: str,
                            num_workers: int=4,
                            batch_size: int=1,
                            shuffle_size: int=100,
                            extract_keys: list[str] | None=None,
                            n_samples_per_epoch: int=3000,
                            modality_names: tuple[str, ...] = ('m1', 'm2'),
                            encoded_img_padder: "callable | None"=None,
                            ) -> tuple[WebDataset, WebLoader]:
    state =  PartialState()
    
    # degraded modality names
    assert len(modality_names) == 2, 'modality_names must be a tuple of length 2'
    degraded_modality_names = tuple([m + '_degraded' for m in modality_names])
    clean_modality_names = tuple([m + '_clean' for m in modality_names])
    
    # ddp splitter
    ddp = state.use_distributed
    if ddp:
        nodesplitter = shardlists.split_by_node
    else:
        nodesplitter = shardlists.single_node_only
    
    # use some keys in tar files
    if extract_keys is not None:
        def extract_keys_fn(sample):
            return {k: sample[k] for k in extract_keys}
    else:
        extract_keys_fn = None
        
    # json file loader
    if 'info.json' in extract_keys:
        def instruction_text_loader(sample):
            sample['info.json'] = sample['info.json']['instruction']
            return sample
    else:
        instruction_text_loader = None
        
    # image transform
    def image_transform(sample):
        for m in (clean_modality_names + degraded_modality_names):
            if m in sample:
                sample[m + '.jpg'] = sample[m + '.jpg'] / 255.0
        return sample
    
    # encoded instruction transform
    def encoded_instruction_transform(sample):
        sample['encoded_instruction.npy'] = sample['encoded_instruction.npy'].squeeze(0)
        return sample
    
    # wrap to a dict to return
    def wrap_to_dict(sample):
        return {k.split('.')[0]: sample[k] for k in extract_keys}

    # make datasets and dataloader    
    ds = WebDataset(dataset_pattern,
                    nodesplitter=nodesplitter,
                    empty_check=False,
                    verbose=True)
    
    # get needed keys
    if extract_keys_fn is not None:
        ds = ds.map(extract_keys_fn)
    ds = ds.decode('torch')
    
    # get instruction text
    if instruction_text_loader is not None:
        ds = ds.map(instruction_text_loader)
        
    # get encoded instruction
    if 'encoded_instruction.npy' in extract_keys:
        ds = ds.map(encoded_instruction_transform)
        
    # get clean images
    if any(m in extract_keys for m in clean_modality_names):
        ds = ds.map(image_transform)
    
    # shuffle and wrap to a dict
    if shuffle_size > 0:
        ds = ds.shuffle(shuffle_size)
    ds = ds.map(wrap_to_dict)
    
    # dataloader
    if num_workers > 0:
        prefetch_factor = 6
        persistent_workers = True
    else:
        prefetch_factor = None
        persistent_workers = False
    
    dl = WebLoader(ds,
                   batch_size=batch_size,
                   num_workers=num_workers,
                   pin_memory=True,
                   prefetch_factor=prefetch_factor,
                   persistent_workers=persistent_workers,)
    if n_samples_per_epoch > 0:
        dl = dl.with_epoch(n_samples_per_epoch)
        dl.with_length(n_samples_per_epoch)
    
    return ds, dl

if __name__ == "__main__":
    ds, dl = get_degraded_wds_loader(dataset_pattern="/Data3/cao/ZiHanCao/exps/panformer/task_datasets/VIF-MSRS/Degraded/{0000..0001}.tar", 
                                     extract_keys=["info.json", "encoded_instruction.npy", 
                                                   "encoded_m1_degraded.npy", "encoded_m2_degraded.npy", 
                                                   "encoded_gt.npy", "m1_clean.jpg", 
                                                   "m2_clean.jpg", "gt.jpg",
                                                   "m1_degraded.jpg", "m2_degraded.jpg"], 
                                     batch_size=16)

    print(f'set length: {len(dl)}')
    
    for sample in dl:
        print(sample)
        break
