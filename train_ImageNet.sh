#!/bin/bash


### BFP - Training

# python train.py \
# --dataset imagenet --datapath /data/cv/ImageNet \
# --model resnet_BFP --q_type BFP \
# --trainer_config ./trainer_config_BFP2.json \
# --device cuda:0 \
# --save test_ImageNet
# --wandb_project OBAQ_new


### FP32 - Training

python train.py \
--dataset imagenet --datapath /data/cv/ImageNet --input_size 224 \
--model resnet --q_type FP32 \
--batch_size 128 --epochs 120 --workers 16 \
--trainer_config ./trainer_config_FP32.json \
--device cuda:0 \
--save test_ImageNet