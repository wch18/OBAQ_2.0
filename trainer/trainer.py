import torch
import torch.nn as nn
from utils import meters

class Trainer:
    '''
    Trainer 
    '''
    def __init__(self, model, scheduler, q_scheduler, train=True):
        ### 
        self.model = model
        self.scheduler = scheduler
        self.q_scheduler = q_scheduler
        ###
        