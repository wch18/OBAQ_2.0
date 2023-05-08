import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction
import torch.nn.functional as F
import numpy as np 

from .Q_core import *
from .Q_params import Q_params
from torch.utils.cpp_extension import load
import time
import os
import sys

sys.path.append("./utils")
from meters import AverageMeter

idx = 0

cur_dir = os.path.abspath(os.path.curdir)

cudnn_convolution = load(name='cudnn_convolution', sources=['./exts/cudnn_convolution.cpp'], verbose=True)

class BFP_conv2d(InplaceFunction):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, q_params:Q_params=Q_params(), quantize_grad=True):
        ctx.input = input
        ctx.weight = weight
        ctx.bias = bias
        ctx.args = stride, padding, dilation, groups
        ctx.q_params = q_params
        ctx.quantize_grad = quantize_grad

        if q_params.state == 'reg':
            output = F.conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            device = output.device
            q_params.computations['W'] = np.product(weight.shape) // 32 * output.shape[-1] * output.shape[-2] # Computation_W = Cin * Cout * K * K * Hout * Wout
            q_params.computations['bA'] = np.product(weight.shape) // 32 * input.shape[-1] * input.shape[-2]   # Computation_bA = Cin * Cout * K * K * Hin * Win
            q_params.C_W = output.shape[2]
            # print('W:', q_params.C_W, np.sqrt(q_params.computations['W']/np.product(weight.shape)*32))
            q_params.C_bA = np.sqrt(weight.shape[0]) * weight.shape[2]
            # print('bA:', q_params.C_bA, np.sqrt(q_params.computations['bA']/np.product(input.shape))*64)

            W_BFPshape = get_BFP_shape(weight.shape, q_params.block_size['W'])
            q_params.sensitivity['W'] = torch.zeros(size=W_BFPshape, device=device)
            bA_BFPshape = get_BFP_shape(input.shape, q_params.block_size['bA'])
            q_params.sensitivity['bA'] = torch.zeros(size=bA_BFPshape, device=device)
            A_BFPshape = bA_BFPshape
            q_params.sensitivity['A'] = torch.zeros(size=A_BFPshape, device=device)
            G_BFPshape = get_BFP_shape(output.shape, q_params.block_size['G'])
            q_params.sensitivity['G'] = torch.zeros(size=G_BFPshape, device=device)

            return output
        
        # return F.conv2d(input, weight, bias,
        #                 stride=stride, padding=padding, dilation=dilation, groups=groups)
        A_sparsity_counter = q_params.sparsity_counter['A']
        W_sparsity_counter = q_params.sparsity_counter['W']

        A_block_size, A_block_bw = q_params.block_size['A'], q_params.int_bwmap['A']
        forward_q_input = BFPQuant(input, A_block_size, A_block_bw, sparsity_counter=A_sparsity_counter)
        
        # forward weight quantization
        W_block_size, W_block_bw = q_params.block_size['W'], q_params.int_bwmap['W']
        forward_q_weight = BFPQuant(weight, W_block_size, W_block_bw, sparsity_counter=W_sparsity_counter)
        
        # forward calculation
        output = F.conv2d(forward_q_input, forward_q_weight, bias,
                        stride=stride, padding=padding, dilation=dilation, groups=groups)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        q_params:Q_params = ctx.q_params
        input = ctx.input
        weight = ctx.weight
        stride, padding, dilation, groups = ctx.args
        input_size = input.shape
        weight_size = weight.shape

        G_sparsity_counter = q_params.sparsity_counter['G']
        bA_sparsity_counter = q_params.sparsity_counter['bA']
            
        # grad activation quantization 
        if ctx.quantize_grad:
            G_block_size, G_block_bw = q_params.block_size['G'], q_params.int_bwmap['G']
            q_grad_output = BFPQuant(grad_output, G_block_size, G_block_bw, True, sparsity_counter=G_sparsity_counter)
        else:
            q_grad_output = grad_output

        ### - Backward Stage - grad_output * weight -> grad_input 
        # backward activation quantization
        bA_block_size, bA_block_bw = q_params.block_size['bA'], q_params.int_bwmap['bA']
        backward_q_input = BFPQuant(input, bA_block_size, bA_block_bw, sparsity_counter=bA_sparsity_counter)
        
        # backward weight quantization
        W_block_size, W_block_bw = q_params.block_size['W'], q_params.int_bwmap['W']
        backward_q_weight = BFPQuant(weight, W_block_size, W_block_bw)

        grad_input = cudnn_convolution.convolution_backward_input(input_size, backward_q_weight, q_grad_output, 
                                                                  stride, padding, dilation, groups,
                                                                  True, False, False)

        ### - Weight/Bias Update Stage - grad_output * input -> grad_weight
        if ctx.needs_input_grad[1]:
            grad_weight = cudnn_convolution.convolution_backward_weight(backward_q_input, weight_size, q_grad_output, 
                                                                        stride, padding, dilation, groups,
                                                                        True, False, False)
        else:
            grad_weight = None

        if ctx.bias is not None and ctx.needs_input_grad[2]:
            grad_bias = q_grad_output.sum([0, 2, 3])
        else:
            grad_bias = None
        
        ### Sensitivity Analysis
        if q_params.state == 'train':
            W_sensitivity = Sensitivity_Analysis(data=weight, grad=grad_weight, block_size=W_block_size, C=q_params.C_W)
            q_params.sensitivity['W'] += W_sensitivity
            bA_sensitivity = Sensitivity_Analysis(data=input, grad=grad_input, block_size=bA_block_size, C=q_params.C_bA)
            q_params.sensitivity['bA'] += bA_sensitivity

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class BFP_linear(InplaceFunction):
    @staticmethod
    def forward(ctx, input, weight, bias=None, q_params:Q_params=Q_params(), quantize_grad=True):
        ctx.input = input
        ctx.weight = weight
        ctx.bias = bias
        ctx.q_params = q_params
        ctx.quantize_grad = quantize_grad

        expand_input = input.unsqueeze(-1).unsqueeze(-1)
        expand_weight = weight.unsqueeze(-1).unsqueeze(-1)

        if q_params.state == 'reg':
            output = F.linear(input, weight, bias)
            device = output.device
            expand_output = output.unsqueeze(-1).unsqueeze(-1)

            q_params.computations['W'] = np.product(expand_weight.shape) // 32 * output.shape[-1] # C_W = Cin * Cout * K * K * Hout * Wout
            q_params.computations['bA'] = np.product(expand_input.shape) // 32 * input.shape[-1]   # C_A = Cin * Cout * K * K *Hin * Win

            W_BFPshape = get_BFP_shape(expand_weight.shape, q_params.block_size['W'])
            q_params.sensitivity['W'] = torch.zeros(size=W_BFPshape, device=device)
            bA_BFPshape = get_BFP_shape(expand_input.shape, q_params.block_size['bA'])
            q_params.sensitivity['bA'] = torch.zeros(size=bA_BFPshape, device=device)
            A_BFPshape = bA_BFPshape
            q_params.sensitivity['A'] = torch.zeros(size=A_BFPshape, device=device)
            G_BFPshape = get_BFP_shape(expand_output.shape, q_params.block_size['G'])
            q_params.sensitivity['G'] = torch.zeros(size=G_BFPshape, device=device)

            return output
        
        # return F.linear(input, weight, bias) 
        
        A_sparsity_counter = q_params.sparsity_counter['A']
        W_sparsity_counter = q_params.sparsity_counter['W']

        # forward activation quantization
        A_block_size, A_block_bw = q_params.block_size['A'], q_params.int_bwmap['A']
        forward_q_input = BFPQuant(expand_input, A_block_size, A_block_bw, sparsity_counter=A_sparsity_counter)
        forward_q_input = forward_q_input.squeeze(-1).squeeze(-1)

        # forward weight quantization
        W_block_size, W_block_bw = q_params.block_size['W'], q_params.int_bwmap['W']
        forward_q_weight = BFPQuant(expand_weight, W_block_size, W_block_bw, sparsity_counter=W_sparsity_counter)
        forward_q_weight = forward_q_weight.squeeze(-1).squeeze(-1)

        forward_q_bias = bias
        
        output = F.linear(forward_q_input, forward_q_weight, forward_q_bias)    

        return output

    @staticmethod
    def backward(ctx, grad_output):
        q_params:Q_params = ctx.q_params
        input = ctx.input
        weight = ctx.weight

        G_sparsity_counter = q_params.sparsity_counter['G']
        bA_sparsity_counter = q_params.sparsity_counter['bA']
        # print(bA_sparsity_counter.val)
    
        # grad activation quantization:
        if ctx.quantize_grad:
            expand_grad_output = grad_output.unsqueeze(-1).unsqueeze(-1)
            G_block_size, G_block_bw = q_params.block_size['G'], q_params.int_bwmap['G']
            q_grad_output = BFPQuant(expand_grad_output, G_block_size, G_block_bw, True, sparsity_counter=G_sparsity_counter)
            q_grad_output = q_grad_output.squeeze(-1).squeeze(-1)
        else:
            q_grad_output = grad_output
        # backward activation quantization
        expand_input = input.unsqueeze(-1).unsqueeze(-1)
        bA_block_size, bA_block_bw = q_params.block_size['bA'], q_params.int_bwmap['bA']
        backward_q_input = BFPQuant(expand_input, bA_block_size, bA_block_bw, sparsity_counter=bA_sparsity_counter)
        backward_q_input = backward_q_input.squeeze(-1).squeeze(-1)
        
        # backward weight quantization
        expand_weight = weight.unsqueeze(-1).unsqueeze(-1)
        W_block_size, W_block_bw = q_params.block_size['W'], q_params.int_bwmap['W']
        backward_q_weight = BFPQuant(expand_weight, W_block_size, W_block_bw)

        backward_q_weight = backward_q_weight.squeeze(-1).squeeze(-1)

        # Backward Stage: 
        C_in = backward_q_input.shape[-1]
        C_out = grad_output.shape[-1]
        q_grad_output_flatten = q_grad_output.view(-1, C_out)
        backward_q_input_flatten = backward_q_input.view(-1, C_in)
        if ctx.needs_input_grad[0]:
            grad_input = q_grad_output_flatten.mm(backward_q_weight)
        else:
            grad_input = None

        # Weight/Bias Update Stage
        if ctx.needs_input_grad[1]:
            grad_weight = q_grad_output_flatten.t().mm(backward_q_input_flatten)
        else:
            grad_weight = None

        if ctx.bias is not None and ctx.needs_input_grad[2]:
            grad_bias = q_grad_output_flatten.sum(0)
        else:
            grad_bias = None
        
        # if q_params.state == 'train':
            W_sensitivity = Sensitivity_Analysis(expand_weight, grad_weight.unsqueeze(-1).unsqueeze(-1), block_size=W_block_size, C=q_params.C_W)
            q_params.sensitivity['W'] += W_sensitivity
            bA_sensitivity = Sensitivity_Analysis(expand_input, grad_input.unsqueeze(-1).unsqueeze(-1), block_size=bA_block_size, C=q_params.C_bA)
            q_params.sensitivity['bA'] += bA_sensitivity

        return grad_input, grad_weight, grad_bias, None, None



# class int_conv2d(InplaceFunction):
#     @staticmethod   
#     def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, bw=[8, 8, 8, 8]): # A,W,G,GA
#         ctx.input = input
#         ctx.weight = weight
#         ctx.bias = bias
#         ctx.args = stride, padding, dilation, groups
#         ctx.bw = bw 
#         q_input= INTQuant(input, bw[0])
#         q_weight= INTQuant(weight, bw[1])
#         # q_input= FPQuant(input)
#         # q_weight= FPQuant(weight)
#         # print(weight[:4,:4,0,0], q_weight[:4,:4,0,0])
#         output = F.conv2d(q_input, q_weight, bias, stride, padding, dilation, groups)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         stride, padding, dilation, groups = ctx.args
#         bw = ctx.bw
#         raw_input = ctx.input
#         raw_weight = ctx.weight
#         input_size = raw_input.shape
#         weight_size = raw_weight.shape
#         # grad_output.clamp_(-0.5, 0.5)
#         q_grad_output= INTQuant(grad_output, bw[2], True)
#         q_input= INTQuant(raw_input, bw[3])
#         q_weight = INTQuant(raw_weight, bw[1])
#         # q_grad_output= FPQuant(grad_output)
#         # q_input= FPQuant(raw_input)
#         # q_weight = FPQuant(raw_weight)

#         grad_input = cudnn_convolution.convolution_backward_input(input_size, q_weight, q_grad_output, 
#                                                                 stride, padding, dilation, groups,
#                                                                 True, False, False)
#         grad_weight = cudnn_convolution.convolution_backward_weight(q_input, weight_size, q_grad_output,
#                                                                 stride, padding, dilation, groups,
#                                                                 True, False, False)
#         norm_grad_weight = grad_weight.norm()
#         if norm_grad_weight >= 1:
#             grad_weight.div_(norm_grad_weight)
#         if ctx.bias is not None:
#             grad_bias = q_grad_output.sum([0, 2, 3])
#         else:
#             grad_bias = None
#         return grad_input, grad_weight, grad_bias, None, None, None, None, None

# class int_linear(InplaceFunction):
#     @staticmethod
#     def forward(ctx, input, weight, bias=None, bw=[8,8,8,8,8]):
#         ctx.input = input
#         ctx.weight = weight
#         ctx.bias = bias
#         ctx.bw = bw
#         q_input= INTQuant(input, bw[0])
#         q_weight= INTQuant(weight, bw[1])
#         q_bias= INTQuant(bias, bw[2])
#         # q_input= FPQuant(input)
#         # q_weight= FPQuant(weight)
#         # q_bias= FPQuant(bias)
#         output = F.linear(q_input, q_weight, q_bias)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         bw = ctx.bw
#         raw_input = ctx.input
#         raw_weight = ctx.weight
#         # grad_output.clamp_(-0.5, 0.5)
#         q_grad_output = INTQuant(grad_output, bw[3], True)
#         q_input= INTQuant(raw_input, bw[4])
#         q_weight = INTQuant(raw_weight, bw[1])

#         C_in = q_input.shape[-1]
#         C_out = grad_output.shape[-1]
#         q_grad_output_flatten = q_grad_output.view(-1, C_out)
#         # q_grad_output_16bit_flatten = q_grad_output_16bit.view(-1, C_out)
#         q_input_flatten = q_input.view(-1, C_in)
#         grad_input = q_grad_output_flatten.mm(q_weight)
#         grad_weight = q_grad_output_flatten.t().mm(q_input_flatten)
#         grad_bias = q_grad_output_flatten.sum(0)
#         norm_grad_weight = grad_weight.norm()
#         if norm_grad_weight >= 1:
#             grad_weight.div_(norm_grad_weight)
#         return grad_input, grad_weight, grad_bias, None

# class INTQConv2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, bias=False, bw=[8,8,8,8]):
#         super(INTQConv2d, self).__init__(in_channels, out_channels, kernel_size,
#                                       stride, padding, dilation, groups, bias)
#         self.bw = bw
        
#     def forward(self, input):
#         output = int_conv2d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.bw)
#         # with torch.no_grad():
#         #     output_2 = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         #     print(diff(output, output_2))
#         return output

# class INTQLinear(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True, bw=[8,8,8,8,8]):
#         super(INTQLinear, self).__init__(in_features, out_features, bias)
#         self.bw = bw

#     def forward(self, input):
#         output = int_linear.apply(input, self.weight, self.bias, self.bw)
#         # output = F.linear(input, self.weight, self.bias)
#         return output

