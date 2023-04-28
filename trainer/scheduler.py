import torch
import torch.nn as nn
import torch.optim as optim

from scheme import Scheme

class Scheduler:
    '''
    Training Scheduler Module:

    '''
    def __init__(self, optimizer:optim.Optimizer, scheme:Scheme, 
                 batches_per_epoch, cur_batch=0, cur_epoch=0) -> None:
        self.optimizer = optimizer
        self.scheme = scheme
        self.lr = self.scheme.init_lr
        
        self.batched_per_epoch = batches_per_epoch
        self.warm_up_rate = 1/(batches_per_epoch * self.scheme.warm_up_epoch)
        
        self.cur_batch = cur_batch
        self.cur_epoch = cur_epoch

        self.train_stage = 0


    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, epoch=None):
        # step and update lr
        if epoch is not None:   # update epoch
            self.cur_epoch = epoch
        else:                   # update batch
            self.cur_batch += 1
            self.update_lr()

    def update_lr(self):
        if self.cur_epoch < self.scheme.warm_up_epoch:
            # warm up stage
            self.lr = self.scheme.init_lr * self.warm_up_rate * self.cur_batch

        if self.scheme.lr_tuning_method == 'Specify':
            # specific lr function
            self.lr = self.scheme.lr_func(self.cur_epoch)
        elif self.scheme.lr_tuning_method == 'Step':
            # Step lr tuning
            if self.cur_epoch == self.scheme.lr_tuning_points[self.train_stage]:
                self.lr *= self.scheme.lr_tuning_rates[self.train_stage]
                self.train_stage += 1
        elif self.scheme.lr_tuning_method == 'Exp':
            # Exp lr tuning
            pass
        elif self.scheme.lr_tuning_method == 'Cos':
            # Cos lr tuning
            pass
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    