#!/bin/bash

CUDA_VISIBLE_DEVICES=[1] python train.py \
--dataset=cifar100 --model=resnet_BFP  --q_type=BFP \
--trainer_config=./trainer_config.json \
--device=cuda:0