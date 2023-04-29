import torch
import torch.nn as nn
from .Q_scheme import Q_Scheme

class Q_Scheduler:
    '''
    Quantization Scheduler Module:
    
    '''
    def __init__(self, q_optimizer=None, q_scheme:Q_Scheme=None, cur_epoch=0) -> None:
        self.q_optimizer = q_optimizer
        self.q_scheme = q_scheme
        self.cur_epoch = cur_epoch

    def step(self):
        self.cur_epoch += 1

        # optimizer update
        if self.cur_epoch % self.q_scheme.update_period == 0:
            self.q_optimizer.update()
    
    def register(self):
        # config
        self.q_optimizer.bwmap_smooth = self.q_scheme.bwmap_smooth
        self.q_optimizer.K_update_mode = self.q_scheme.K_update_mode
        self.q_optimizer.target_bit_W = self.q_scheme.target_bit_W
        self.q_optimizer.target_bit_bA = self.q_scheme.target_bit_bA

        for q_params in self.q_optimizer.q_params_list:
            for datatype in ['A', 'W', 'G', 'bA']:
                q_params.set_int_bwmap(datatype=datatype, int_bwmap=self.q_scheme.init_bwmap[datatype])

    def train(self):
        pass

    def eval(self):
        pass

