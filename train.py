import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import argparse
import os
import wandb
from .data.dataset import get_dataset

parser = argparse.ArgumentParser()


### 
parser.add_argument('--results_dir', default='./results', help='results dir')
parser.add_argument('--save', default='', help='saved folder')

### dataset arguments
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--datadir', type=str, default='/home/wch/data/cifar100')

### model arguments
parser.add_argument('--model', default='resnet', choices=['resnet', 'resnet_BFP'])
parser.add_argument('--input_size', type=int, default=None)
parser.add_argument('--model_config', default='')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--workers', type=int, default=8)

### trainer arguments
parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', default='SGD')
