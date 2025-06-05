# CUDA_VISIBLE_DEVICES=1 \
# python val.py --img 1280 \
# --weights yolov5_visible.pt \
# --data LLVIP.yaml --iou-thres 0.5

# CUDA_VISIBLE_DEVICES=0 \
# python val.py --img 1280 \
# --weights yolov5l.pt \
# --data M3FD.yaml

# CUDA_VISIBLE_DEVICES=1 python val.py --img 1280 --weights yolov5l.pt --data M3FD.yaml

CUDA_VISIBLE_DEVICES=1 \
DATASET_NAME=MSRS \
python val.py --img 1280 \
--weights yolov5l.pt \
--data MSRS.yaml \
--conf-thres 0.3 \
--iou-thres 0.3