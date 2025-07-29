## panRWKV_v3
# T_MAX="2048" \
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch \
# --gpu_ids "1" \
# --config_file configs/huggingface/accelerate.yaml \
# accelerate_inference_on_fusion.py \
# -c "configs/panRWKV_config.yaml" \
# -m "RWKVFusion_v12.RWKVFusion" \
# --val_bs 1 --dataset msrs \
# --dataset_mode 'all' \
# --extra_save_name '11_30_all' \
# --model_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/RWKVFusion_v12_RWKVFusion/vis_ir_joint/2024-11-08-14-44-12_panRWKV_sb45j3w5_vis_ir_joint_rwkv5_2_wo_omnishift_lerp_factor=0/weights/ema_model.pth/model.safetensors" \
# --pad_window_base 32 \
# --only_y \
# --normalize

# --analysis_fused \
# --debug
# --extra_save_name 'v2' \
# --extra-save-name msrs_rs_tno_joint_v2 \
# --load-spec-key 'ema_model'

## MGDN
# NCCL_P2P_LEVEL="NVL" \
# NCCL_P2P_DISABLE="1" \
# NCCL_IB_DISABLE="1" \
# OMP_NUM_THREADS="6" \
# python -u -m accelerate.commands.launch \
# --gpu_ids "0" \
# --config_file configs/huggingface/accelerate.yaml \
# accelerate_inference_on_fusion.py \
# -c "configs/MGDN_config.yaml" \
# -m "MGDN.MGFF" \
# --val_bs 1 --dataset m3fd \
# --dataset_mode 'test' \
# --model_path "log_file/MGDN_MGFF/msrs/2024-11-19-23-02-55_MGDN_usom2r99_MGDN config on MSRS dataset base model/weights/checkpoints/checkpoint_1/model.safetensors" \
# --extra_save_name 'direct_train_v2'

T_MAX="65536" \
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
python -u -m accelerate.commands.launch \
--gpu_ids "7" \
--config_file configs/huggingface/accelerate.yaml \
accelerate_inference_on_fusion.py \
-c "configs/panRWKV_config.yaml" \
-m "RWKVFusion_v12.RWKVFusion" \
--val_bs 1 --dataset msrs \
--dataset_mode 'all' \
--extra_save_name '11_30_all' \
--model_path "/Data3/cao/ZiHanCao/exps/panformer/log_file/RWKVFusion_v12_RWKVFusion/vis_ir_joint/2024-11-08-14-44-12_panRWKV_sb45j3w5_vis_ir_joint_rwkv5_2_wo_omnishift_lerp_factor=0/weights/ema_model.pth/model.safetensors" \
--pad_window_base 32 \
--only_y \
--normalize
