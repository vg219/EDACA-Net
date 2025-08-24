# TORCH_DISTRIBUTED_DEBUG="DETAIL" \
CUDA_VISIBLE_DEVICES="0,1" \
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
accelerate launch \
--config_file configs/huggingface/accelerate.yaml \
--num_processes 1 \
--gpu_ids "0" \
--main_process_port 29506 \
accelerate_run_maskgit_binary_engine.py \
--config maskgit \
--dataset_name dif_msrs \
--comment "maskgit"
# accelerate_run_maskgit_engine.py \
