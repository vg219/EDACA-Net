CUDA_VISIBLE_DEVICES="0,1" \
NCCL_P2P_LEVEL="NVL" \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
OMP_NUM_THREADS="6" \
TORCH_DISTRIBUTED_DEBUG="DETAIL" \
accelerate launch \
--config_file configs/huggingface/accelerate.yaml \
--multi_gpu \
--num_processes 2 \
--gpu_ids "0,1" \
--main_process_port 29506 \
accelerate_lfq_vae_engine.py \
--config lfq_vae_config.yaml \
--comment "loop_up_free_quantization_vae" \
--resume_ckpt "/Data3/cao/ZiHanCao/exps/panformer/log_file/lfq_vae/unify_image_fusion_vae/2024-11-17-02-24-00_lfq_vae_loop_up_free_quantization_vae/weights/ema_model.pth"
