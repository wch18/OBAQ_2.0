import torch
import torch.nn as nn

class Scheduler:
    '''
    Training Scheduler Module:

    '''
    def __init__(self, optimizer, scheme, cur_epoch) -> None:
        self.optimizer = optimizer
        self.scheme = scheme
        self.cur_epoch = cur_epoch

    def step(self):
        self.cur_epoch += 1
        