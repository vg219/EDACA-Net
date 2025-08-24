import os
import lpips

dataset = 'RoadSceneFusion'
methods = [
#     'ydtr_TNO',
#     'dcformer_u2fusion_5_2_10_TNO',
    # 'new_U2Fusion'
    # 'PMGI',
    # 'NSST',
    # 'IFCNN-MAX_IV',
    # 'GTF',
    # 'densefuse_addition_100',
    # 'DDcGAN'
    # 'RFN_NEST_TNO'
    # 'dcformer_RS_TNO_u2fusion_loss_wx_new'
    
    # 'dcformer_u2fusion',
    # 'densefuse_reproduce',
    # 'GTF',
    # 'ifcnn',
    # 'nsct',
    # 'u2fusion',
    # 'ydtr'
    # 'defuse'
    # 'rfnnest'
    # 'new_U2Fusion_100'
    # 'DDcGAN_500_optim_205'
    # 'dcformer_RS_TNO_u2fusion_loss_wx_new'
    # 'SwinFuse',
    # 'SwinFusion'
    "LRRNet"
]
test_ir_path = f'./data/{dataset}/test/ir'
test_vi_path = f'./data/{dataset}/test/vi'

for m in methods:
    print(m)
    fuse_path = f'./2_DL_Result/{dataset}/{m}/'
    
    # os.system(
    #     f'fidelity --fid --kid --lpips --input1 {fuse_path} --input2 {test_ir_path} --batch-size 1 --kid-subset-size 10'
    # )
    
    # os.system(
    #     f'fidelity --fid --kid --input1 {fuse_path} --input2 {test_vi_path} --batch-size 1 --kid-subset-size 10'
    # )
    os.system(
        f'python lpips_2dirs.py -d0 {fuse_path} -d1 {test_ir_path} -o {m}_ir.txt'
    )
    os.system(
        f'python lpips_2dirs.py -d0 {fuse_path} -d1 {test_vi_path} -o {m}_vi.txt'   
    )
    
    print('-'*20)
    
    