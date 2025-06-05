# RANK="1" LOCAL_RANK="1" WORLD_SIZE="2" MASTER_ADDR="localhost" MASTER_PORT="5700" 
CUDA_VISIBLE_DEVICES="1" \
python train.py \
--img 1024 \
--weights "yolov5/yolov5l.pt" \
--data M3FD.yaml \
--epochs 300 \
--name M3FD_export \
--batch 16 \
--label-smoothing 0. \
--save-period 20 \
--device "1"