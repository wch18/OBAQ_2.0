import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datetime import datetime

import argparse
import os
import sys
import wandb
import time
import json
from data.dataset import get_dataset, fast_collate
from data.preprocess import get_transform
# from models.resnet_BFP import resnet_BFP
from models import resnet, resnet_BFP

from models.Q_modules import Q_Optimizer
from trainer import Trainer, get_optimizer, Scheduler, Q_Scheduler, Scheme, Q_Scheme, WandbLogger
import torch._dynamo
torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser()

### global arguments
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--trainer_config', type=str, default=None)
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--log_mode', type=str, default='debug')
parser.add_argument('--channels-last', type=bool, default=False)
# parser.add_argument('--ddp', type=bool, default=False)

### logging arguments
parser.add_argument('--results_dir', default='./results', help='results dir')
parser.add_argument('--save', default='', help='saved folder')
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--save_model_mode', default='best', help='save model mode')

### dataset arguments
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--datapath', type=str, default='/home/wch/data/cifar100')

### model arguments
parser.add_argument('--model', default='resnet_BFP', choices=['resnet', 'resnet_BFP'])
parser.add_argument('--input_size', type=int, default=32)
parser.add_argument('--model_config', default='')
parser.add_argument('--q_type', default=None)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--workers', type=int, default=8)

### trainer arguments
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', default='SGD')
parser.add_argument('--warm_up_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1, help='init lr')

### BFPQ argument
parser.add_argument('--target_bit_W', type=int, default=2)
parser.add_argument('--target_bit_bA', type=int, default=2)
parser.add_argument('--K_update_mode', type=str, default='BinarySearch')



def main(args):
    is_main_process = args.local_rank == 0
    output_target = sys.stdout if is_main_process else open(os.devnull, 'w')
    print('Global Setting...', file=output_target)
    args.ddp = int(os.getenv('WORLD_SIZE', 0))>1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    day_stamp = datetime.now().strftime('%Y-%m-%d')
    time_stamp = datetime.now().strftime('%H:%M:%S')

    if args.ddp:
        print('Start Distributed Dataparallel Processing...', file=output_target)
        args.device = torch.cuda.device(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.gpus = dist.get_world_size()

        print('Work on {} GPUs'.format(args.gpus), file=output_target)

    if args.save == '':
        save_folder = args.model + '_' + time_stamp
    else:
        save_folder = args.save

    save_path = args.results_dir + '/' + day_stamp + '/' + save_folder

    if not os.path.exists(save_path) and is_main_process:
        os.makedirs(save_path, exist_ok=True)

    if is_main_process:
        print('Save at ', save_path, file=output_target)

        with open(save_path + '/args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        wandb_log = args.wandb_project is not None

        wandb_mode = "disabled" if args.wandb_project else "online"

        if wandb_log:
            if args.trainer_config is not None:
                wandb_config = {'config_save':save_path}
            else:
                wandb_config = args.__dict__

            print('Wandb Setting (Optional) ...', file=output_target)
            wandb.init(project=args.wandb_project, 
                       mode=wandb_mode,
                       name=save_folder, 
                       config=wandb_config)
            
    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format


    print('-------- Data Loading ---------', file=output_target)
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
    # collate_fn = lambda b: fast_collate(b, memory_format)
    collate_fn = None
    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=True, prefetch_factor=4)
    
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, prefetch_factor=4)
    
    print('--------- Model Creating ---------',file=output_target)

    # model = resnet_BFP(depth=18, dataset='cifar100').to(args.device)
    # model = resnet_BFP(depth=18, dataset=args.dataset).to(args.device)
    model = resnet(depth=50, dataset=args.dataset).cuda().to(memory_format=memory_format)
    # compile_model = torch.compile(model, mode='reduce-overhead')
    # model = compile_model
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = get_optimizer(args.optimizer, model.parameters())
    scheme = Scheme(init_lr=args.lr, warm_up_epoch=args.warm_up_epoch)
    scheduler = Scheduler(optimizer=optimizer, scheme=scheme,
                          batches_per_epoch=len(train_loader))
    
    if args.q_type == 'BFP':
        q_optimizer = Q_Optimizer(q_params_list=model.q_params_list())
    else:
        q_optimizer = None

    q_scheme = Q_Scheme(q_type=args.q_type, target_bit_bA=args.target_bit_bA, target_bit_W=args.target_bit_W,
                        K_update_mode=args.K_update_mode)
    q_scheduler = Q_Scheduler(q_optimizer=q_optimizer, q_scheme=q_scheme,
                              batches_per_epoch=len(train_loader))
    # q_scheduler = None

    
    print('Trainer Creating...',file=output_target)
    if wandb_log:
        wandb_logger = WandbLogger()
    else:
        wandb_logger = None
    trainer = Trainer(model=model,
                    scheduler=scheduler, q_scheduler=q_scheduler,
                    criterion=criterion,
                    train_loader=train_loader, test_loader=test_loader,
                    device=args.device, log_freq=args.log_freq,
                    wandb_logger=wandb_logger)

    if args.trainer_config is not None:
        trainer.load_config(args.trainer_config)

    dummy_input = torch.zeros([args.batch_size, 3, args.input_size, args.input_size], device=args.device)
    trainer.register(dummy_input=dummy_input)

    trainer.save_config(save_dir=save_path)

    print('-------- Training --------',file=output_target)
    best_prec = 0
    for epoch in range(args.epochs):
        print('Train Epoch\t:', epoch,file=output_target)
        trainer.train(epoch)
        trainer.train_logger.log('END TRAIN')
        print('Test Epoch:\t', epoch,file=output_target)
        trainer.test(epoch)
        trainer.train_logger.log('END TEST')

        if args.save_model_mode == 'best':
            if trainer.train_logger.top1.avg >= best_prec:
                model_dir = save_path + '/best'
                trainer.save_model(model_dir)
        elif 'every' in args.save_model_mode:
            freq = int(args.save_model_mode.split('_')[1])
            if epoch % freq == 0:
                model_dir = save_path + '/epoch_' + str(epoch)
                trainer.save_model(model_dir)
        else:
            pass
        
        best_prec = max(best_prec, trainer.train_logger.top1.avg)
        print('current_best_prec: ', best_prec,file=output_target)

    print('--------- Training Done ---------',file=output_target)
    print(best_prec)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)