# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --num_processes 1 \
# --gpu_ids "0" \
# --config_file configs/huggingface/accelerate.yaml \
# accelerate_inference_on_sharpening.py \
# -c configs/panRWKV_config.yaml \
# -m RWKVFusion_v12.RWKVFusion \
# --val_bs 1 \
# --dataset pavia \
# --model_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/RWKVFusion_v12_RWKVFusion/pavia/2024-11-07-13-13-13_panRWKV_waubyiui_pavia_rwkv5_2_wo_omnishift/weights/ema_model.pth/model.safetensors" \
# --split_patch \
# --save_mat \
# --full_res \

# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --num_processes 1 \
# --gpu_ids "0" \
# --config_file configs/huggingface/accelerate.yaml \
# accelerate_inference_on_sharpening.py \
# -c configs/FeINFN_config.yaml \
# -m FeINFN.FeINFNet \
# --val_bs 1 \
# --dataset qb \
# --model_path "/home/YuJieLiang/Efficient-MIF-back-master-6-feat/log_file/FeINFN_FeINFNet/qb/2025-01-10-15-49-26_FeINFN_aefr9lic_FeINFN on qb dataset/weights/ema_model.pth/model.safetensors" \
# --save_mat 
# --full_res 
# --split_patch 

# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --num_processes 1 \
# --gpu_ids "6" \
# --config_file configs/huggingface/accelerate.yaml \
# accelerate_inference_on_sharpening.py \
# -c configs/ENACIR_config.yaml \
# -m ENACIR.ENACIR \
# --val_bs 1 \
# --dataset harvard_x4 \
# --model_path "/data2/users/yujieliang/exps/Efficient-MIF-back-master-6-feat/log_file/ENACIR_ENACIR/harvard_x4/2025-06-17-22-37-46_ENACIR_u76rdqoo_ENACIR on cave_x4 dataset/weights/ema_model.pth/model.safetensors" \
# --save_mat \
# --split_patch

# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# accelerate launch \
# --num_processes 1 \
# --gpu_ids "6" \
# --config_file configs/huggingface/accelerate.yaml \
# accelerate_inference_on_sharpening.py \
# -c configs/ENACIR_config.yaml \
# -m ENACIR.ENACIR \
# --val_bs 1 \
# --dataset pavia \
# --model_path "/data2/users/yujieliang/exps/Efficient-MIF-back-master-6-feat/log_file/ENACIR_ENACIR/pavia/2025-06-18-16-29-50_ENACIR_nnf4ldd5_ENACIR on pavia dataset/weights/ema_model.pth/model.safetensors" \
# --save_mat 

NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
accelerate launch \
--num_processes 1 \
--gpu_ids "6" \
--config_file configs/huggingface/accelerate.yaml \
accelerate_inference_on_sharpening.py \
-c configs/MHIIF_config.yaml \
-m MHIIF.MHIIF_ \
--val_bs 1 \
--dataset cave_mulit_x16 \
--model_path "/data2/users/yujieliang/exps/Efficient-MIF-back-master-6-feat/log_file/MHIIF_MHIIF_/cave_mulit_x4/2025-08-13-13-35-49_MHIIF_r07a74ga_MHIIF_ms on cave_mulit_x4 dataset/weights/ema_model.pth/model.safetensors" 
