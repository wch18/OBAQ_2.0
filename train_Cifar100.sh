#!/bin/bash

python train.py \
--dataset cifar100 --datapath /datashare01/cv/cifar100 \
--model resnet_BFP --q_type BFP \
--trainer_config ./trainer_config.json \
--device cuda:2 \
--save test
# --wandb_project OBAQ_new