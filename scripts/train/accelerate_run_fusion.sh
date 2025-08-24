# # accelerate run
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
--proj_name EDACANet \
-m 'EDACANet.EDACANet' \
-c 'EDACANet_config.yaml' \
--dataset pavia \
--num_worker 6 -e 1000 --train_bs 4 --val_bs 1 \
--aug_probs 0.0 0.0 --loss l1ssim --grad_accum_steps 1 \
--val_n_epoch 10 \
--checkpoint_every_n 10 \
--comment "EDACANet on pavia dataset" \
--logger_on \
--sanity_check 
