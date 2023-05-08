import torch 
import torch.nn as nn
import numpy as np

from .Q_core import round

class Q_params:
    def __init__(self):
        # A: forward Activation 
        self.layer = None

        self.state = None
        self.C_W = 1
        self.C_bA = 1

        self.block_size = {
            'A': [4,4,1,1],
            'W': [4,4,1,1],
            'G': [4,4,1,1],
            'bA': [4,4,1,1]  
        }

        self.bwmap_new = {
            'A': None,
            'W': None,
            'G': None,
            'bA': None
        }

        self.bwmap = {
            'A': None,
            'W': None,
            'G': None,
            'bA': None
        }

        self.int_bwmap = {
            'A': None,
            'W': None,
            'G': None,
            'bA': None  
        }

        self.computations = {
            'A' : 1,
            'W' : 1,
            'G' : 1,
            'bA': 1,
        }

        self.sensitivity = {
            'A': None,
            'W': None,
            'G': None,
            'bA': None  
        }
        
        self.mask = {
            'A': None,
            'W': None,
            'G': None,
            'bA': None
        }

        self.sparsity_counter = {
            'A' : None,
            'W' : None,
            'G' : None,
            'bA': None,
        }

    def set_grad(self, datatype, grad):
        self.grad[datatype] = grad

    def set_mask(self, datatype, mask):
        self.mask[datatype] = mask

    def set_bwmap(self, datatype, bwmap):
        self.bwmap[datatype] = bwmap

    def set_int_bwmap(self, datatype, int_bwmap):
        if isinstance(int_bwmap, int):
            self.bwmap[datatype] = torch.ones_like(self.sensitivity[datatype]) * int_bwmap
            self.int_bwmap[datatype] = self.bwmap[datatype].int()
        else:
            self.int_bwmap[datatype] = int_bwmap
            
    def update_bwmap(self, datatype, bwmap_smooth=0):
        self.bwmap[datatype] = self.bwmap[datatype] * bwmap_smooth + self.bwmap_new[datatype] * (1-bwmap_smooth)
    
    def update_int_bwmap(self, datatype):
        self.int_bwmap[datatype] = round(self.bwmap[datatype], scale=2, stochastic=False)

    def set_sensitivity(self, datatype, sensitivity, sensitivity_smooth=0):
        self.sensitivity[datatype] = sensitivity

    def acc_sensitivity(self, datatype, sensitivity):
        self.sensitivity[datatype] += sensitivity

    def set_sparsity_counter(self, datatype, counter=None):
        self.sparsity_counter[datatype] = counter
    
    # def update_sensitivity(self, datatype):
    #     if self.grad[datatype] is None:
    #         return 
    #     elif datatype == 'W':
            

    # def bwmap_mean(self, datatype):
    #     self.bwmap[datatype] = 