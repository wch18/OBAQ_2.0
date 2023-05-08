import torch
import torch.nn as nn
import numpy as np
import time

from .Q_core import *
from .Q_params import Q_params
from typing import Sequence

def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

class Q_Optimizer():
    def __init__(self, q_params_list:Sequence[Q_params], 
                 bwmap_smooth:float=0.5, 
                 K_update_mode:str ='BinarySearch',
                 target_bit_W = 2,
                 target_bit_bA = 2) -> None:
        # 
        self.q_params_list = q_params_list    # Q_params <-> parameters
        self.bwmap_smooth = bwmap_smooth
        self.K_update_mode = K_update_mode
        self.target_bit_W = target_bit_W
        self.target_bit_bA = target_bit_bA

        # Local Space
        self.epoch = 0
        self.K_W = 0
        self.K_bA = 0
        self.bwmap_new = []
        self.computations = []

    def update(self):
        # Top level update call
        print('update W')
        # update Weight bwmap
        self.K_W = self.K_update(target_bit=self.target_bit_W, datatype='W')
        self.update_bwmap(datatype='W')
        print('mean_int_bw ', mean_bwmap(self.q_params_list, datatype='W', bwmaptype='int_bwmap')[0])
        print('update bA')
        # update backward Act bwmap
        self.K_bA = self.K_update(target_bit=self.target_bit_bA, datatype='bA')
        self.update_bwmap(datatype='bA')
        print('mean_int_bw ', mean_bwmap(self.q_params_list, datatype='bA', bwmaptype='int_bwmap')[0])

    def zero_sensitivity(self):
        for q_params in self.q_params_list:
            q_params.sensitivity['W'] = torch.zeros_like(q_params.sensitivity['W'])
            q_params.sensitivity['bA'] = torch.zeros_like(q_params.sensitivity['bA'])

    def tuning_sensitivity(self, batches):
        print('batches=', batches)
        for q_params in self.q_params_list:
            q_params.sensitivity['W'] /= batches
            q_params.sensitivity['bA'] /= batches

    def reset_sparsity_counter(self):
        print('reset')
        for q_params in self.q_params_list:
            for datatype in ['A', 'W','G','bA']:
                if q_params.sparsity_counter is not None:
                    q_params.sparsity_counter[datatype].reset()
        
    def get_bwmap_new(self, K, datatype):
        # get latest bwmap with K
        for q_params in self.q_params_list:
            bwmap_with_K = torch.clip(q_params.sensitivity[datatype]-K, 0, 8)
            q_params.bwmap_new[datatype] = bwmap_with_K

    def update_bwmap(self, datatype):
        # update bwmap of all q_param of self.q_params_list
        for q_params in self.q_params_list:
            q_params.update_bwmap(datatype=datatype, bwmap_smooth=self.bwmap_smooth)
            q_params.update_int_bwmap(datatype=datatype)

    def K_init(self, target_bit, datatype):
        S = 0
        N = 0
        for q_params in self.q_params_list:
            sensitivity = q_params.sensitivity[datatype]
            S += torch.sum(sensitivity)
            N += np.product(sensitivity.shape)

        K = float((S/N - target_bit).cpu())
        return K

    def K_update(self, target_bit, datatype):
        bit_dis = 1 # distance of mean_bw and target_bit
        tol = 0.05  # tolerance of max distance
        
        K = self.K_init(target_bit, datatype)

        if self.K_update_mode == 'BinarySearch':
            K_lower, K_upper = K-8, K+8
            while K_lower < K_upper:
                K = (K_lower + K_upper)/2
                self.get_bwmap_new(K=K, datatype=datatype)
                mean_bw, _ = mean_bwmap(self.q_params_list, datatype=datatype, bwmaptype='bwmap_new')
                print(datatype, K, mean_bw)
                bit_dis = mean_bw - target_bit
                if np.abs(bit_dis) < tol:
                    break
                elif bit_dis > 0:
                    K_lower = K + tol/2
                else:
                    K_upper = K - tol/2

        elif self.K_update_mode == 'LERP':
            # Linear 
            pass
    