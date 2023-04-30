import torch
import torch.nn as nn
import sys
import time
sys.path.append('./')
from models.resnet_BFP import resnet_BFP
from models.Q_modules import *
from trainer import Scheme, Q_Scheme, Scheduler, Q_Scheduler, Trainer
from data.dataset import get_dataset
from data.preprocess import get_transform
from torch.utils.data import DataLoader

# from gpu_mem_track import MemTracker

inputs = torch.randn((128, 3, 32, 32)).to('cuda:0')
test_model = nn.Sequential(BFPQConv2d(3, 16, 3, padding=1), BFPQConv2d(16, 32, 3, padding=1)).to('cuda:0')
labels = torch.randn((128, 100)).to('cuda:0')
criterion = nn.MSELoss()

resnet_bfp = resnet_BFP(dataset='cifar100', depth=18).to('cuda:0')

optimizer = torch.optim.SGD(resnet_bfp.parameters(), 0.1)
q_optimizer = Q_Optimizer(resnet_bfp.q_params_list())
scheme = Scheme()
q_scheme= Q_Scheme()
scheduler = Scheduler(optimizer, scheme, 100)
q_scheduler = Q_Scheduler(q_optimizer, q_scheme)

# train_loader = []
# test_loader = []
# for _ in range(100):
#     train_loader.append((torch.randn((128, 3, 32, 32)).to('cuda:0'), torch.randn((128, 100)).to('cuda:0')))

# for _ in range(20):
#     test_loader.append((torch.randn((128, 3, 32, 32)).to('cuda:0'), torch.randn((128, 100)).to('cuda:0')))
dataset = 'cifar100'
datapath = '/home/wch/data/cifar100'
batch_size = 128
train_transform = get_transform(dataset, 
                                input_size=32, augment=True)
test_transform = get_transform(dataset, 
                                input_size=32, augment=False)

train_set = get_dataset(dataset, split='train', 
                        transform=train_transform, 
                        datasets_path=datapath)
test_set = get_dataset(dataset, split='val', 
                        transform=test_transform, 
                        datasets_path=datapath)

train_loader = DataLoader(train_set, 
                            batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)

test_loader = DataLoader(test_set,
                            batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
trainer = Trainer(model=resnet_bfp,
                  scheduler=scheduler, q_scheduler=q_scheduler,
                  criterion=nn.CrossEntropyLoss(),
                  train_loader=train_loader, test_loader=test_loader)

scheduler = Scheduler(optimizer=optimizer, scheme=scheme, batches_per_epoch=391)


trainer.register(inputs)

trainer.train(0)

# while True:
#     outputs = resnet_bfp(inputs)
#     loss = criterion(labels, outputs)
#     loss.backward()
#     # q_optimizer.update()
#     time.sleep(1)