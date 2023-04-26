import torch
import torch.nn as nn
from Q_core import *
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
import time
import numpy as np


# a_block = rearrange(a, 
#                     '(oc_b bs0) (ic_b bs1) (kx_b bs2) (ky_b bs3) -> oc_b ic_b kx_b ky_b bs0 bs1 bs2 bs3', 
#                     bs0=4, bs1=4, bs2=1, bs3=1)
    
# print(a_block.shape)

# a = torch.randn([4,64,3,3]).to('cuda:0')
# st = time.time()

# for i in range(1000):
#     BFPshape = BFP_getshape(a.shape, [4, 4, 1, 1])
#     block_bw = torch.ones(BFPshape).to('cuda:0') * 2
#     a_BFP = BFPQuant(a, [4, 4, 1, 1], block_bw)

# print(time.time()-st)

a = torch.linspace(-3.5, 3.5, 16).reshape(4,4,1,1)
BFPshape = get_BFP_shape(a.shape, [4, 4, 1, 1])
block_bw = torch.ones(BFPshape) * 4
a_BFP = BFPQuant(a, [4, 4, 1, 1], block_bw).reshape(4, 4)
print(F.mse_loss(a.reshape(4,4), a_BFP))