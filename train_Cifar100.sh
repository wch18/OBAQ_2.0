#!/bin/bash

python train.py \
--dataset cifar100 --datapath /home/unbuntu-server/data/ \
--model resnet_BFP --q_type BFP \
--trainer_config ./trainer_config_BFP2.json \
--device cuda:0 \
--save test
# --wandb_project OBAQ_new