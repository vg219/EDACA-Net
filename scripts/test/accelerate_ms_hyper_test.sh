
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
accelerate launch \
--num_processes 1 \
--gpu_ids "6" \
--config_file configs/huggingface/accelerate.yaml \
accelerate_inference_on_sharpening.py \
-c configs/EDACANet_config.yaml \
-m EDACANet.EDACANet \
--val_bs 1 \
--dataset pavia \
--model_path "/ENACIR_nnf4ldd5_ENACIR on pavia dataset/weights/ema_model.pth/model.safetensors" \
--save_mat 
