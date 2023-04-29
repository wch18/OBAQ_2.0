import torch
import torch.nn as nn

from .Q_core import *
from .Q_base_layers import *
from .Q_params import Q_params

class BFPQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BFPQConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.q_params = Q_params()

    def forward(self, input):
        output = BFP_conv2d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.q_params)
        return output
        
class BFPQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BFPQLinear, self).__init__(in_features, out_features, bias)
        self.q_params = Q_params()

    def forward(self, input):
        output = BFP_linear.apply(input, self.weight, self.bias, self.q_params)
        return output