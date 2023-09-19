import torch
import torch.nn as nn

from .Q_core import *
from .Q_base_layers import *
from .Q_params import Q_params

class BFPQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, q_params=None):
        super().__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.q_params = q_params if q_params else Q_params()

    def forward(self, input):
        output = BFP_conv2d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.q_params)
        return output
        
class BFPQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, q_params=None):
        super().__init__(in_features, out_features, bias)
        self.q_params = q_params if q_params else Q_params()

    def forward(self, input):
        output = BFP_linear.apply(input, self.weight, self.bias, self.q_params)
        return output
    
class INTQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, bw=[8,8,8,8]):
        super().__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.bw = bw
        
    def forward(self, input):
        output = INT_conv2d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.bw)
        return output

class INTQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bw=[8,8,8,8,8]):
        super().__init__(in_features, out_features, bias)
        self.bw = bw

    def forward(self, input):
        output = INT_linear.apply(input, self.weight, self.bias, self.bw)
        # output = F.linear(input, self.weight, self.bias)
        return output
    
class FPQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        
    def forward(self, input):
        output = FP_conv2d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

class FPQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, input):
        output = FP_linear.apply(input, self.weight, self.bias)
        return output