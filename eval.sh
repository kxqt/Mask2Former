#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export DETECTRON2_DATASETS=/data

# python train_net.py \
#   --config-file configs/coco/panoptic-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml \
#   --eval-only MODEL.WEIGHTS checkpoints/coco/panoptic_seg/swin_tiny_mask2former.pkl
  
python train_net.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_vit_base_bs16_50ep.yaml \
  --num-gpus 1