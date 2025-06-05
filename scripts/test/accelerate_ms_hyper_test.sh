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

NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
accelerate launch \
--num_processes 1 \
--gpu_ids "0" \
--config_file configs/huggingface/accelerate.yaml \
accelerate_inference_on_sharpening.py \
-c configs/FeINFN_config.yaml \
-m FeINFN.FeINFNet \
--val_bs 1 \
--dataset qb \
--model_path "/home/YuJieLiang/Efficient-MIF-back-master-6-feat/log_file/FeINFN_FeINFNet/qb/2025-01-10-15-49-26_FeINFN_aefr9lic_FeINFN on qb dataset/weights/ema_model.pth/model.safetensors" \
--save_mat 
# --full_res 
# --split_patch 