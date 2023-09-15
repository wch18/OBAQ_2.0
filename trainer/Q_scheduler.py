import torch
import torch.nn as nn
from utils.meters import AverageMeter
from .Q_scheme import Q_Scheme

class Q_Scheduler:
    '''
    Quantization Scheduler Module:
    
    '''
    def __init__(self, q_optimizer=None, q_scheme:Q_Scheme=None, 
                 batches_per_epoch=1, cur_epoch=0) -> None:
        self.q_optimizer = q_optimizer
        self.q_scheme = q_scheme
        self.batches_per_epoch = batches_per_epoch
        self.cur_epoch = cur_epoch

    def zero_sensitivity(self):
        if self.q_scheme.q_type == 'BFP':
            self.q_optimizer.zero_sensitivity()
        else:
            pass
        
    def step(self):
        self.cur_epoch += 1
        # optimizer update
        if self.q_scheme.q_type == 'BFP':
            if self.cur_epoch % self.q_scheme.update_period == 0:
                self.q_optimizer.tuning_sensitivity(self.batches_per_epoch*self.q_scheme.update_period)
                self.q_optimizer.update()
                self.q_optimizer.reset_sparsity_counter()
        else:
            pass
    
    def register(self):
        # config
        if self.q_scheme.q_type == 'BFP':
            self.q_optimizer.bwmap_smooth = self.q_scheme.bwmap_smooth
            self.q_optimizer.K_update_mode = self.q_scheme.K_update_mode
            self.q_optimizer.target_bit_W = self.q_scheme.target_bit_W
            self.q_optimizer.target_bit_bA = self.q_scheme.target_bit_bA
            
            for q_params in self.q_optimizer.q_params_list:
                for datatype in ['A', 'W', 'G', 'bA']:
                    q_params.set_int_bwmap(datatype=datatype, int_bwmap=self.q_scheme.init_bwmap[datatype])
                    q_params.set_sparsity_counter(datatype=datatype, counter=AverageMeter())
        else:
            pass