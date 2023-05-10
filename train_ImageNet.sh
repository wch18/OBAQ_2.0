CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29556 train.py \
--save test_DDP \
--dataset imagenet \
--model resnet_BFP \
--model_config "{'depth': 18}" \
--workers 8 \
--b 120 \
--epochs 95 \
--warm_up_epoch 5