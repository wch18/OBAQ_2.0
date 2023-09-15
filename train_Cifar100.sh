#!/bin/bash

python train.py \
--dataset=cifar100 --model=resnet_BFP  --q_type=BFP \
--trainer_config ./trainer_config.json \
--device cuda:0 \
--save update_3_new \
--wandb_project OBAQ_new