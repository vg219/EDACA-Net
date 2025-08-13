# # accelerate run
# CUDA_VISIBLE_DEVICES="2" \
# T_MAX="65536" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --main_process_port 29506 \
# accelerate_run_main.py \
# --proj_name MHIIF_J2 \
# -m 'MHIIF_J2.MHIIF_J2' \
# -c 'MHIIF_config.yaml' \
# --dataset cave_mulit_x4 \
# --num_worker 0 -e 500 --train_bs 4 --val_bs 1 \
# --aug_probs 0.0 0.0 --loss hermite --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --checkpoint_every_n 10 \
# --comment "MHIIF_J2 on cave_mulit_x4 dataset" \
# --logger_on \
# --sanity_check 
# --resume_path "/data2/users/yujieliang/exps/Efficient-MIF-back-master-6-feat/log_file/MHIIF_J_MHIIF_J/cave_x4/2025-07-22-11-04-06_MHIIF_7dwi9pbg_MHIIF_J on cave_x4 dataset/weights/checkpoints/checkpoint_108" \

# CUDA_VISIBLE_DEVICES="3" \
# T_MAX="65536" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --main_process_port 29506 \
# accelerate_run_main.py \
# --proj_name MHIIF_J2_Hermite \
# -m 'MHIIF_J2_Hermite.MHIIF_J2_Hermite' \
# -c 'MHIIF_config.yaml' \
# --dataset cave_x4 \
# --num_worker 0 -e 2000 --train_bs 128 --val_bs 1 \
# --aug_probs 0.0 0.0 --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --checkpoint_every_n 10 \
# --comment "MHIIF_J2_Hermite on cave_x4 dataset" \
# --logger_on 
# --sanity_check 

CUDA_VISIBLE_DEVICES="0" \
T_MAX="65536" \
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
accelerate launch \
--config_file configs/huggingface/accelerate.yaml \
--main_process_port 29506 \
accelerate_run_main.py \
--proj_name MHIIF_rbf \
-m 'MHIIF_rbf.MHIIF_rbf' \
-c 'MHIIF_config.yaml' \
--dataset harvard_mulit_x4 \
--num_worker 0 -e 2000 --train_bs 4 --val_bs 1 \
--aug_probs 0.0 0.0 --loss l1ssim --grad_accum_steps 1 \
--val_n_epoch 10 \
--checkpoint_every_n 10 \
--comment "MHIIF_rbf on harvard_mulit_x4 dataset" \
--logger_on \
--sanity_check 

## accelerate run
# CUDA_VISIBLE_DEVICES="5" \
# T_MAX="65536" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --main_process_port 29506 \
# accelerate_run_main.py \
# --proj_name ENACIR \
# -m 'ENACIR_V2.ENACIR_V2' \
# -c 'ENACIR_config.yaml' \
# --dataset cave_x4 \
# --num_worker 0 -e 2200 --train_bs 4 --val_bs 1 \
# --aug_probs 0.0 0.0 --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --checkpoint_every_n 10 \
# --comment "ENACIR_V2 on cave_x4 dataset" \
# --logger_on \
# --sanity_check \


# CUDA_VISIBLE_DEVICES="6" \
# T_MAX="65536" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --main_process_port 29506 \
# accelerate_run_main.py \
# --proj_name ENACIR \
# -m 'ENACIR.ENACIR' \
# -c 'ENACIR_config.yaml' \
# --dataset chikusei \
# --num_worker 0 -e 2200 --train_bs 4 --val_bs 1 \
# --aug_probs 0.0 0.0 --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --checkpoint_every_n 10 \
# --comment "ENACIR on chikusei dataset" \
# --logger_on \
# --sanity_check \
# CUDA_VISIBLE_DEVICES="0" \
# T_MAX="65536" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --main_process_port 29506 \
# accelerate_run_main.py \
# --proj_name FeINFN \
# -m 'FeINFN.FeINFNet' \
# -c 'FeINFN_config.yaml' \
# --dataset qb \
# --num_worker 6 -e 800 --train_bs 64 --val_bs 1 \
# --aug_probs 0.0 0.0 --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --checkpoint_every_n 10 \
# --comment "FeINFN on qb dataset" \
# --logger_on 
# --sanity_check
# MHIIF
# CUDA_VISIBLE_DEVICES="0" \
# T_MAX="65536" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --main_process_port 29506 \
# accelerate_run_main.py \
# --proj_name MHIIF \
# -m 'MHIIF.MHIIF_' \
# -c 'MHIIF_config.yaml' \
# --dataset harvard_mulit_x4 \
# --num_worker 6 -e 2000 --train_bs 4 --val_bs 1 \
# --aug_probs 0.0 0.0 --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --checkpoint_every_n 10 \
# --comment "MHIIF_ms on harvard_mulit_x4 dataset" \
# --logger_on \
# --sanity_check

# CUDA_VISIBLE_DEVICES="2" \
# T_MAX="65536" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --main_process_port 29506 \
# accelerate_run_main.py \
# --proj_name MHIIF \
# -m 'hermite.hermite' \
# -c 'MHIIF_config.yaml' \
# --dataset cave_x4 \
# --num_worker 6 -e 2000 --train_bs 4 --val_bs 1 \
# --aug_probs 0.0 0.0 --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --checkpoint_every_n 10 \
# --comment "MHIIF_hermite on cave_x4 dataset" \
# --logger_on 
# --resume_path "/home/YuJieLiang/Efficient-MIF-back-master-6-feat/log_file/hermite_hermite/cave_x4/2025-03-03-14-13-20_MHIIF_gabyanlo_MHIIF_hermite on cave_x4 dataset/weights/checkpoints/checkpoint_71" 


# CUDA_VISIBLE_DEVICES="0" \
# T_MAX="65536" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --main_process_port 29506 \
# accelerate_run_main.py \
# --proj_name MHIIF \
# -m 'MHIIF_g.MHIIF_g' \
# -c 'MHIIF_config.yaml' \
# --dataset cave_x4 \
# --num_worker 6 -e 2000 --train_bs 4 --val_bs 1 \
# --aug_probs 0.0 0.0 --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --checkpoint_every_n 10 \
# --comment "MHIIF_g on cave_x4 dataset" \
# --logger_on \
# --sanity_check 
# --resume_path "/home/YuJieLiang/Efficient-MIF-back-master-6-feat/log_file/MHIIF_g_MHIIF_g/cave_x4/2024-12-22-19-53-13_MHIIF_2dpkn5vq_MHIIF_g on cave_x4 dataset/weights/checkpoints/checkpoint_55" 




## accelerate run
# CUDA_VISIBLE_DEVICES="0" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch --config_file configs/huggingface/accelerate.yaml accelerate_main.py \
# --proj_name panRWKV_v3 --arch panRWKV --sub_arch v3 --dataset wv3 \
# --num_worker 6 -e 800 -b 32 --aug_probs 0. 0. --loss l1ssim --grad_accum_steps 2 \
# --checkpoint_every_n 20 --val_n_epoch 20  \
# --comment "panRWKV config on wv3 dataset model" \
# --log_metric \
# --logger_on \
# --resume_path "/Data2/ZiHanCao/exps/panformer/weight/2024-04-17-02-07-23_panRWKV_9sqz9900/ep_800"
# --pretrain_model_path "/Data2/ZiHanCao/exps/panformer/weight/2024-04-15-20-24-17_panRWKV_96z6zd29/panRWKV_96z6zd29.pth" \
# --non_load_strict


## accelerate test
# CUDA_VISIBLE_DEVICES="1,2" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch --gpu_ids "0,1" \
# --multi_gpu --num_processes 2 accelerate_test.py

## panRWKV_v3
# CUDA_VISIBLE_DEVICES="0" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch \
# --config_file configs/huggingface/accelerate.yaml \
# accelerate_run_main.py \
# --proj_name panRWKV_v3 \
# --arch panRWKV \
# --sub_arch v3 \
# --dataset wv3 \
# --num_worker 0 -e 800 --train_bs 64 --val_bs 1 \
# --aug_probs 0. 0. --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --comment "panRWKV config on WV3 dataset tiny model without q_shift" \
# --checkpoint_every_n 10 \
# --metric_name_for_save "SAM" \
# --log_metric --logger_on \
# --pretrain_model_path "/Data3/cao/ZiHanCao/exps/panformer/weight/2024-06-10-20-12-19_panRWKV_z9ydu64u/panRWKV_z9ydu64u.pth" \
# --non_load_strict
# --resume_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/panRWKV_v3/gf2/2024-07-28-21-59-11_panRWKV_nsruqx06_panRWKV config on GF2 dataset tiny model/weights/ep_10"
# --sanity_check \


# panRWKV_v3 MEF
# CUDA_VISIBLE_DEVICES="0" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch \
# --config_file configs/huggingface/accelerate.yaml \
# accelerate_run_main.py \
# --proj_name panRWKV_v3 \
# --arch panRWKV \
# --sub_arch v3 \
# --dataset vis_ir_joint \
# --num_worker 0 -e 800 --train_bs 18 --val_bs 1 \
# --aug_probs 0.0 0. --loss drmffusion --grad_accum_steps 2 \
# --val_n_epoch -1 \
# --comment "panRWKV config on vis_ir_joint dataset tiny model" \
# --checkpoint_every_n 10 \
# --metric_name_for_save "PSNR" \
# --fusion_crop_size 96 \
# --log_metric --logger_on \
# --pretrain_model_path '/Data3/cao/ZiHanCao/exps/panformer/log_file/panRWKV_v3/vis_ir_joint/2024-07-28-15-39-39_panRWKV_v64628cj_panRWKV config on vis_ir_joint dataset tiny model/weights/ep_10/model.safetensors' \


## MGDN
# CUDA_VISIBLE_DEVICES="0" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --gpu_ids 0 \
# accelerate_run_main.py \
# --proj_name MGDN --arch MGDN --dataset msrs \
# -m MGDN.MGFF \
# -c MGDN_config.yaml \
# --num_worker 0 -e 400 --train_bs 14 --val_bs 1 \
# --aug_probs 0. 0. --loss u2fusion --grad_accum_steps 1 \
# --val_n_epoch -1  \
# --comment "MGDN config on MSRS dataset base model" \
# --log_metric \
# --checkpoint_every_n 10 \
# --logger_on \
# --metric_name_for_save "PSNR" \


## panRWKV_v4 pansharpening and HMIF
# CUDA_VISIBLE_DEVICES="0,1,2" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch \
# --config_file configs/huggingface/accelerate.yaml \
# --multi_gpu \
# --num_processes 3 \
# --gpu_ids "0,1,2" \
# accelerate_run_main.py \
# --proj_name panRWKV_v8_cond_norm \
# --arch panRWKV \
# --sub_arch v8_cond_norm \
# --dataset qb \
# --num_worker 0 -e 800 --train_bs 48 --val_bs 1 \
# --aug_probs 0. 0. --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --comment "k=2 with q shift conditional scale shift gated for attn and ffn with parallel MIFM" \
# --metric_name_for_save "SAM" \
# --log_metric --logger_on \
# --pretrain_model_path "/Data2/ZiHanCao/exps/panformer/log_file/panRWKV_v8_cond_norm/gf2/2024-08-09-03-09-26_panRWKV_u4o64esy_k=2 with q shift conditional scale shift gated for attn and ffn with parallel MIFM/weights/ema_model.pth" \
# --non_load_strict
# --checkpoint_every_n None\


## panRWKV_v10 VIS IR joint
# CUDA_VISIBLE_DEVICES="0,1" \
# T_MAX="8192" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch \
# --config_file configs/huggingface/accelerate.yaml \
# --num_processes 1 \
# --gpu_ids "0" \
# accelerate_run_main.py \
# --proj_name RWKVFusion_v10_muti_modal.py \
# -m 'RWKVFusion_v10_multi_modal.RWKVFusion' \
# -c 'panRWKV_config.yaml' \
# --dataset med_harvard \
# --num_worker 0 -e 600 --train_bs 6 --val_bs 1 \
# --aug_probs 0. 0. --loss drmffusion --grad_accum_steps 2 \
# --val_n_epoch 1 \
# --comment "RWKVFusion_v10 multi-modal input with llm feature on MSRS dataset" \
# --checkpoint_every_n 5 \
# --logger_on \
# --pad_window_base 32 \
# --pretrain_model_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/RWKVFusion_v10_multi_modal.RWKVFusion/med_harvard/2024-09-12-22-50-46_panRWKV_799uy9ph_RWKVFusion_v10 multi-modal input with llm feature on MSRS dataset/weights/ema_model.pth/pytorch_model.bin"
# --resume_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/RWKVFusion_v10_multi_modal.RWKVFusion/vis_ir_joint/2024-09-11-22-53-27_panRWKV_vl7ev29i_RWKVFusion_v10 multi-modal input with llm feature on MSRS dataset/weights/checkpoints/checkpoint_9"
# --pretrain_model_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/RWKVFusion_v10_multi_modal.RWKVFusion/wv3/2024-09-08-23-22-25_panRWKV_ldgt6uou_RWKVFusion_v10 multi-modal input with llm feature/weights/ema_model.pth"
# --fast_eval_n_samples 80 \


## panRWKV_v10 pansharpening
# CUDA_VISIBLE_DEVICES="0,1" \
# T_MAX="65536" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --multi_gpu \
# --num_processes 2 \
# --gpu_ids "0,1" \
# --main_process_port 29506 \
# accelerate_run_main.py \
# --proj_name RWKVFusion_v12 \
# -m 'RWKVFusion_v12.RWKVFusion' \
# -c 'panRWKV_config.yaml' \
# --dataset realmff+mff_whu \
# --num_workers 6 -e 50 --train_bs 12 --val_bs 1 \
# --aug_probs 0.0 0.0 --loss drmffusion --grad_accum_steps 2 \
# --val_n_epoch 10 \
# --checkpoint_every_n 10 \
# --comment "realmff_rwkv5_2" \
# --logger_on \
# --only_y_train \
# --ckpt_max_limit 10 \
# --resume_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/RWKVFusion_v12_RWKVFusion/realmff/2024-11-21-20-09-07_panRWKV_qjnhhyqz_realmff_rwkv5_2_wo_omnishift_lerp_factor=0/weights/checkpoints/checkpoint_84"

# --debug
# --resume_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/RWKVFusion_v12_RWKVFusion/med_harvard/2024-11-09-18-15-55_panRWKV_v5rpgg3a_med_harvard_rwkv5_2_wo_omnishift_lerp_factor=0/weights/checkpoints/checkpoint_361" \
# --pretrain_model_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/RWKVFusion_v12_RWKVFusion/vis_ir_joint/2024-11-07-22-39-42_panRWKV_e7ao9d2e_vis_ir_joint_rwkv5_2_wo_omnishift/weights/ema_model.pth/model.safetensors"



# Unfolding DEQ
# CUDA_VISIBLE_DEVICES="0,1" \
# T_MAX="65536" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --multi_gpu \
# --num_processes 2 \
# --gpu_ids "0,1" \
# --main_process_port 29506 \
# accelerate_run_main.py \
# --proj_name Unfolding_DEQ \
# -m 'unfold_deq_pan.Unfolding_DEQ' \
# -c 'unfolding_deq_config.yaml' \
# --dataset wv3 \
# --num_worker 6 -e 800 --train_bs 64 --val_bs 1 \
# --aug_probs 0.0 0.0 --loss l1ssim --grad_accum_steps 1 \
# --val_n_epoch 10 \
# --checkpoint_every_n 10 \
# --comment "unfolding_deq on wv3 dataset" \
# --logger_on \


#* SwinFusion
# CUDA_VISIBLE_DEVICES="0,1" \
# T_MAX="65536" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --config_file configs/huggingface/accelerate.yaml \
# --num_processes 1 \
# --gpu_ids "1" \
# --main_process_port 29506 \
# accelerate_run_main.py \
# --proj_name SwinFusion \
# -m 'swinfusion.SwinFusion' \
# -c 'swinfusion_config.yaml' \
# --dataset med_harvard \
# --num_workers 6 -e 800 --train_bs 8 --val_bs 1 \
# --aug_probs 0.0 0.0 --loss u2fusion --grad_accum_steps 2 \
# --val_n_epoch 10 \
# --checkpoint_every_n 10 \
# --comment "med_harvard_swinfusion" \
# --logger_on \
# --ckpt_max_limit 10 \
# --only_y_train \
# --resume_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/swinfusion_SwinFusion/med_harvard/2024-11-22-16-36-10_SwinFusion_w59ekzqg_med_harvard_swinfusion/weights/checkpoints/checkpoint_3"