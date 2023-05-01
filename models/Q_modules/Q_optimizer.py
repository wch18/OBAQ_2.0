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
        self.temp_bwmap = []
        self.computations = []

    def update(self):
        # Top level update call
        print('update W')
        # update Weight bwmap
        self.K_W = self.K_update(target_bit=self.target_bit_W, datatype='W')
        self.update_bwmap(datatype='W')
        print('mean_int_bw ', self.mean_bw(datatype='W', bwmaptype='int_bwmap'))
        print('update bA')
        # update backward Act bwmap
        self.K_bA = self.K_update(target_bit=self.target_bit_bA, datatype='bA')
        self.update_bwmap(datatype='bA')
        print('mean_int_bw ', self.mean_bw(datatype='bA', bwmaptype='int_bwmap'))

    def zero_sensitivity(self):
        for q_params in self.q_params_list:
            q_params.sensitivity['W'] = torch.zeros_like(q_params.sensitivity['W'])
            q_params.sensitivity['bA'] = torch.zeros_like(q_params.sensitivity['bA'])

    def tuning_sensitivity(self, batches):
        print('batches=', batches)
        for q_params in self.q_params_list:
            q_params.sensitivity['W'] /= batches
            q_params.sensitivity['bA'] /= batches

    def mean_bw(self, datatype=None, bwmaptype='temp_bwmap'):
        # return the mean of bwmap:temp_bwmap, int_bwmap or smooth bwmap

        layer_mean_bw = []
        layer_computations = []

        if bwmaptype == 'temp_bwmap':
            for bwmap in self.temp_bwmap:
                layer_mean_bw.append(np.average(bwmap.cpu().numpy()))
            layer_computations = self.computations
        elif bwmaptype == 'int_bwmap':
            for q_params in self.q_params_list:
                bwmap = q_params.int_bwmap[datatype]
                layer_mean_bw.append(np.average(bwmap.cpu().numpy()))
                layer_computations.append(q_params.computations[datatype])
        elif bwmaptype == 'bwmap':
            for q_params in self.q_params_list:
                bwmap = q_params.bwmap[datatype]
                layer_mean_bw.append(np.average(bwmap.cpu().numpy()))
                layer_computations.append(q_params.computations[datatype])

        mean_bw = np.average(layer_mean_bw, weights=layer_computations)
        return mean_bw

    def get_temp_bwmap(self, K, datatype):
        # get latest bwmap with K

        self.temp_bwmap = []
        self.computations = []

        for q_params in self.q_params_list:
            bwmap_with_K = torch.clip(q_params.sensitivity[datatype]-K, 0, 8)
            self.temp_bwmap.append(bwmap_with_K)
            self.computations.append(q_params.computations[datatype])

    def update_bwmap(self, datatype):
        # update bwmap of all q_param of self.q_params_list
        cur_bwmap = 0
        for q_params in self.q_params_list:
            q_params.update_bwmap(datatype=datatype, bwmap_new=self.temp_bwmap[cur_bwmap], bwmap_smooth=self.bwmap_smooth)
            q_params.update_int_bwmap(datatype=datatype)
            cur_bwmap += 1

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
                self.get_temp_bwmap(K=K, datatype=datatype)
                mean_bw = self.mean_bw(bwmaptype='temp_bwmap')
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
    