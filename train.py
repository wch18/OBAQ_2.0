import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from datetime import datetime

import argparse
import os
import wandb
import time
from data.dataset import get_dataset
from data.preprocess import get_transform
from models.resnet_BFP import resnet_BFP

# from .trainer.trainer import Trainer, get_optimizer
# from .trainer.scheme import Scheme
from models.Q_modules import Q_Optimizer
from trainer import Trainer, get_optimizer, Scheduler, Q_Scheduler, Scheme, Q_Scheme

parser = argparse.ArgumentParser()

### global arguments
parser.add_argument('--seed', type=int, default=123, help='random seed')

### logging arguments
parser.add_argument('--results_dir', default='./results', help='results dir')
parser.add_argument('--save', default='', help='saved folder')
parser.add_argument('--log_freq', type=int, default=10)

### dataset arguments
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--datapath', type=str, default='/home/wch/data/cifar100')

### model arguments
parser.add_argument('--model', default='resnet', choices=['resnet', 'resnet_BFP'])
parser.add_argument('--input_size', type=int, default=32)
parser.add_argument('--model_config', default='')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--workers', type=int, default=8)

### trainer arguments
parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', default='SGD')
parser.add_argument('--warm_up_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1, help='init lr')

### BFPQ argument
# parser.add_argument('--K_search_mode', type=)

def main(args):
    print('Global Setting...')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if args.save == '':
        save_folder = args.model + '_seed_' + str(args.seed)
    else:
        save_folder = args.save

    save_path = args.results_dir + '/' + args.save
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    print('-------- Data Loading ---------')
    train_transform = get_transform(args.dataset, 
                                    input_size=args.input_size, augment=True)
    test_transform = get_transform(args.dataset, 
                                   input_size=args.input_size, augment=False)
    
    train_set = get_dataset(args.dataset, split='train', 
                            transform=train_transform, 
                            datasets_path=args.datapath)
    test_set = get_dataset(args.dataset, split='val', 
                           transform=test_transform, 
                           datasets_path=args.datapath)
    
    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=True)
    
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    
    print('--------- Model Creating ---------')

    model = resnet_BFP(depth=18, dataset='cifar100').to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = get_optimizer(args.optimizer, model.parameters())
    scheme = Scheme(init_lr=args.lr, warm_up_epoch=args.warm_up_epoch)
    scheduler = Scheduler(optimizer=optimizer, scheme=scheme,
                          batches_per_epoch=len(train_loader))
    q_optimizer = Q_Optimizer(q_params_list=model.q_params_list())
    q_scheme = Q_Scheme()
    q_scheduler = Q_Scheduler(q_optimizer=q_optimizer, q_scheme=q_scheme,
                              batches_per_epoch=len(train_loader))


    
    print('Trainer Creating...')
    trainer = Trainer(model=model,
                      scheduler=scheduler, q_scheduler=q_scheduler,
                      criterion=criterion,
                      train_loader=train_loader, test_loader=test_loader,
                      device=args.device, log_freq=args.log_freq)
    dummy_input = torch.zeros([args.batch_size, 3, args.input_size, args.input_size], device=args.device)
    trainer.register(dummy_input=dummy_input)

    print('-------- Training --------')
    best_prec = 0
    for epoch in range(args.epochs):
        print('Train Epoch\t:', epoch)
        trainer.train(epoch)
        trainer.training_log.log('END TRAIN')
        print('Test Epoch:\t', epoch)
        trainer.test(epoch)
        trainer.training_log.log('END TEST')
        best_prec = max(best_prec, trainer.training_log.top1.avg)

    print('--------- Training Done ---------')
    print(best_prec)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)