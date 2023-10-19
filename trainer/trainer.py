import torch
import numpy as np
import torch.optim as optim
from utils import meters
from .scheduler import Scheduler
from .Q_scheduler import Q_Scheduler
from .logger import BasicLogger, WandbLogger
from data.dataset import data_prefetcher
import wandb
import time
import json
import os
import sys

def get_optimizer(optimizer_name, params):
    if optimizer_name == 'SGD':
        return optim.SGD(params=params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == 'Adam':
        return optim.Adam(params=params)
    
class Trainer:
    '''
    Trainer:
    '''
    def __init__(self, model, scheduler:Scheduler, q_scheduler:Q_Scheduler, criterion, 
                 train_loader, test_loader, device='cuda:0', 
                 train_logger:BasicLogger=BasicLogger(), log_freq=10,
                 wandb_logger:WandbLogger=WandbLogger(), output_target=sys.stdout):
        ### 
        self.model = model
        self.scheduler = scheduler  
        self.q_scheduler = q_scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.train_logger = train_logger
        self.log_freq = log_freq
        self.wandb_logger = wandb_logger
        self.output_target = output_target

        ### 

    def register(self, dummy_input):
        self.train_logger.output_target = self.output_target
        if self.wandb_logger is not None:
            self.wandb_logger.train_logger = self.train_logger
            self.wandb_logger.q_optimizer = self.q_scheduler.q_optimizer
        
        if self.q_scheduler.q_scheme.q_type != 'BFP':
            self.q_scheduler.q_optimizer = None
            return
        
        self.model.register()
        self.model(dummy_input)
        self.q_scheduler.register()

        print('Register Done.', file=self.output_target)

    def train(self, epoch):
        self.forward(epoch=epoch, dataloader=self.train_loader, train=True)

    def test(self, epoch):
        self.forward(epoch=epoch, dataloader=self.test_loader, train=False)

    def forward(self, epoch, dataloader, train=True):
        
        if train:
            self.model.train(train)
            self.q_scheduler.zero_sensitivity()
        else:
            self.model.eval()

        self.train_logger.reset()
        
        time_stamp = time.time()

        for batch, (inputs, labels) in enumerate(dataloader):
            
            self.train_logger.data_time.update(time.time()-time_stamp)
            time_stamp = time.time()

            self.scheduler.zero_grad()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            
            if train:
                loss.backward()
                self.scheduler.step()

            prec = meters.accuracy(outputs.detach(), labels, (1, 5))
            
            self.train_logger.batch_time.update(time.time()-time_stamp)
            time_stamp = time.time()

            self.train_logger.losses.update(loss.detach().cpu())
            self.train_logger.top1.update(prec[0])
            self.train_logger.top5.update(prec[1])
            
            if batch % self.log_freq == 0:
                self.train_logger.log(batch)

        if self.wandb_logger is not None:
            self.wandb_logger.update(epoch=epoch, train=train)
            self.wandb_logger.log()

        if train:
            self.scheduler.update_epoch()
            self.q_scheduler.step()
        else:
            self.train_logger.update()

    

        print('Epoch: {}, lr: {}'.format(epoch, self.scheduler.lr), file=self.output_target)

    def save_config(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        trainer_config_path = save_dir + '/trainer_config.json'
        with open(trainer_config_path, 'w') as f:
            train_config = {
                'Common': getattr(self.scheduler.scheme, '__dict__', None),
                'Quantization':getattr(self.q_scheduler.q_scheme, '__dict__', None)
            }
            json.dump(train_config, f, indent=4)
            print('Successful Dumping Trainer Config to '+ trainer_config_path, file=self.output_target)

    def load_config(self, trainer_config_path):
        with open(trainer_config_path, 'r') as f:
            trainer_config = json.load(f)
            print('Successful Loading Trainer Config from ' + trainer_config_path, file=self.output_target)
            setattr(self.scheduler.scheme, '__dict__', trainer_config['Common'])
            setattr(self.q_scheduler.q_scheme, '__dict__', trainer_config['Quantization'])

    def save_state(self, save_dir):
        trainer_state = save_dir + '/trainer_state.npy' 

    def load_state(self, trainer_state_path):
        trainer_state = trainer_state_path

    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        model_dict_path = model_dir + '/model.pth'
        torch.save(self.model.state_dict(), model_dict_path)
        if self.q_scheduler.q_scheme.q_type == 'BFP':
            q_params_dict_path = model_dir + '/q_params.npy'
            np.save(q_params_dict_path, self.model.q_params_dict())
        print('Successful Saving Model to ' + model_dir + ' ...', file=self.output_target)

    def load_model(self, model_dir):
        model_dict_path = model_dir + '/model.pth'
        self.model.load_state_dict(torch.load(model_dict_path))
        if self.q_scheduler.q_scheme.q_type == 'BFP':
            q_params_dict_path = model_dir + '/q_params.npz'
            self.model.load_q_params_dict(np.load(q_params_dict_path))
