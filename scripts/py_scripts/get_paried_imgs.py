import numpy as np
from pathlib import Path
from PIL import Image
from shutil import copyfile
import re

fused_path = Path('/Data3/cao/ZiHanCao/ImageFusionBenchmarks/MFI-WHU_raw/_raw')
# far_path = Path('/Data3/cao/ZiHanCao/datasets/MFF-WHU/MFI-WHU/source_1')
# near_path = Path('/Data3/cao/ZiHanCao/datasets/MFF-WHU/MFI-WHU/source_2')
# clear_path = Path('/Data3/cao/ZiHanCao/datasets/MFF-WHU/MFI-WHU/full_clear')


# paired_names = []

# i = 0
# for fused_file in fused_path.iterdir():
#     fused_img = np.array(Image.open(fused_file))
#     err_lst = []
#     err_name_lst = []
#     for far_file in clear_path.iterdir():
#         far_img = np.array(Image.open(far_file))
#         if fused_img.shape == far_img.shape:
#             err_lst.append((fused_img - far_img).mean())
#             err_name_lst.append(far_file.name)
        
#     if len(err_lst) > 0:
#         min_err = min(err_lst)
#         min_err_name = err_name_lst[err_lst.index(min_err)]
#         i += 1
#         paired_names.append((fused_file.name, min_err_name))
#         print(f'{i}/{len(list(fused_path.iterdir()))}: {fused_file.name}, {min_err_name}')
#     else:
#         raise ValueError(f'not find paired far image for {fused_file.name}')

#     arr = np.asanyarray(paired_names)
#     np.savetxt('paired_names.txt', arr, fmt='%s', delimiter=' ')



def fused_name_to_int_str_name(fused_name):
    str_num = re.compile(r'(\d+)').findall(fused_name)[0]
    # unzfill str
    fused_name = str(str_num)
    
    return fused_name

def MFF_WHU_benchmark_name_to_dataset_name(path='/Data3/cao/ZiHanCao/ImageFusionBenchmarks/MFI-WHU_raw/paired_names.txt'):
    txt = np.loadtxt(path, dtype=str, delimiter=" ")
    mapping = {txt[i][0]: txt[i][1] for i in range(len(txt))}
    def _map(idx: str):
        return mapping[idx]
    
    return _map

save_path = Path('/Data3/cao/ZiHanCao/ImageFusionBenchmarks/MFF-WHU')
mapping = MFF_WHU_benchmark_name_to_dataset_name()

for fused_method_file in fused_path.iterdir():
    for fused_file in fused_method_file.iterdir():
        fused_name = fused_file.name
        str_idx = fused_name_to_int_str_name(fused_name)
        dataset_name = mapping(str_idx) + '.jpg'
        file_path = save_path / fused_method_file.name / dataset_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print(file_path)
        copyfile(fused_file, file_path)