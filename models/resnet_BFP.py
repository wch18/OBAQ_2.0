import torch.nn as nn
import torchvision.transforms as transforms
import math
import numpy as np
# from .modules.BWPred import BWPred, estimator
from .modules.layer_status import layer_status
from .modules.layers import BFPQConv2d, BFPQLinear
from .modules.BFPQConfig import Qconfig
from .modules.BFPQscheme import BFPQScheme

__all__ = ['resnet_BFP']

def BFPconv3x3(in_planes, out_planes, stride=1):
    return BFPQConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def init_model(model):
    for m in model.modules():
        if isinstance(m, BFPQConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    for m in model.modules():
        if isinstance(m, Bottleneck_BFP):
            nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock_BFP) or isinstance(m, BasicBlock_BFP):
            nn.init.constant_(m.bn2.weight, 0)

    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()

def init_status(model):   # 对每一个conv2d和linear层，配置其量化器
    total_layers = 0
    for layer in model.modules():
        if isinstance(layer, BFPQConv2d) or isinstance(layer, BFPQLinear):
            # status = layer_status(
            #                  block_size=Qconfig.block_size, 
            #                  grad_smooth_gamma=Qconfig.grad_smooth_gamma,
            #                  batch_size=Qconfig.batch_size,
            #                  bw_smooth_beta=Qconfig.bw_smooth_beta,
            #                  K = Qconfig.K,
            #                  device=Qconfig.device)
            # layer.set_status(status)
            layer.status.block_size = Qconfig.block_size
            layer.status.bw_block_size = Qconfig.bw_block_size
            layer.status.grad_smooth_beta = (1/np.e)**(1/Qconfig.grad_smooth_gamma) - 0.01
            layer.status.batch_size = Qconfig.batch_size
            layer.status.bw_smooth_beta = Qconfig.bw_smooth_beta
            layer.status.device = Qconfig.device
            layer.status.init_bwmap(Qconfig.init_bwmap)
            layer.status.stochastic = Qconfig.stochastic
            layer.status.merge_block = Qconfig.merge_block
            layer.status.layer_sparsity = Qconfig.layer_sparsity

            layer.id = total_layers
            total_layers += 1
            layer.status.id = layer.id
            
class BasicBlock_BFP(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_BFP, self).__init__()
        self.conv1 = BFPconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = BFPconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out  

class Bottleneck_BFP(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_BFP, self).__init__()
        self.conv1 = BFPQConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BFPQConv2d(planes, planes, kernel_size=3, stride=stride,
                             padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BFPQConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_BFP(nn.Module):
    def __init__(self):
        super(ResNet_BFP, self).__init__()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BFPQConv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def save_bwmap(self, bwmap_file):
        bwmap_dict = {}
        for name, layer in self.named_modules():
            if isinstance(layer, BFPQConv2d) or isinstance(layer, BFPQLinear):    
                bwmap_dict[name + '_W_bwmap'] = layer.status.bwmap['W']
                bwmap_dict[name + '_GA_bwmap'] = layer.status.bwmap['GA']
                bwmap_dict[name + '_W_latest_bwmap'] = layer.status.latest_bwmap['W']
                bwmap_dict[name + '_GA_latest_bwmap'] = layer.status.latest_bwmap['GA']
                bwmap_dict[name + '_W_int_bwmap'] = layer.status.int_bwmap['W']
                bwmap_dict[name + '_GA_int_bwmap'] = layer.status.int_bwmap['GA']
        np.save(bwmap_file, bwmap_dict)

    def load_bwmap(self, bwmap_file):
        bwmap_dict = np.load(bwmap_file, allow_pickle=True).item()
        for name, layer in self.named_modules():
            if isinstance(layer, BFPQConv2d) or isinstance(layer, BFPQLinear):    
                layer.status.bwmap['W'] = bwmap_dict[name + '_W_bwmap'] 
                layer.status.bwmap['GA'] = bwmap_dict[name + '_GA_bwmap']
                layer.status.latest_bwmap['W'] = bwmap_dict[name + '_W_latest_bwmap']
                layer.status.latest_bwmap['GA'] = bwmap_dict[name + '_GA_latest_bwmap']            

    @staticmethod
    def regularization(model, weight_decay=1e-4):
        l2_params = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                l2_params += m.weight.pow(2).sum()
                if m.bias is not None:
                    l2_params += m.bias.pow(2).sum()
        return weight_decay * 0.5 * l2_params

class ResNet_imagenet_BFP(ResNet_BFP):
    def __init__(self, num_classes=1000,
                 block=Bottleneck_BFP, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet_BFP, self).__init__()
        self.inplanes = 64
        self.conv1 = BFPQConv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = BFPQLinear(512 * block.expansion, num_classes)

        init_model(self)
        init_status(self)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 1e-2},
            {'epoch': 60, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 80, 'lr': 1e-4}
        ]

class ResNet_cifar100_BFP(ResNet_BFP):
    def __init__(self, num_classes=10,
                 block=BasicBlock_BFP, layers=[2, 2, 2, 2]): 
        super(ResNet_cifar100_BFP, self).__init__()
        self.inplanes = 64
        self.conv1 = BFPQConv2d(3, 64, kernel_size=3, stride=1, padding=1,
                             bias=False)
        # self.bn1 = RangeBN(16, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = BFPQLinear(512 * block.expansion, num_classes, bias=True)
        self.diff = None
        init_model(self)
        init_status(self)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 5e-4, 'momentum': 0.9},
            {'epoch': 60, 'lr': 2e-2},
            {'epoch': 120, 'lr': 4e-3},
            {'epoch': 160, 'lr': 8e-4}
        ]

class ResNet_cifar100_BFP_simple(ResNet_BFP):
    def __init__(self, num_classes=10,
                 block=BasicBlock_BFP, depth=18): 
        super(ResNet_cifar100_BFP_simple, self).__init__()
        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.conv1 = BFPQConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                             bias=False)
        # self.bn1 = RangeBN(16, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = BFPQLinear(64, num_classes, bias=True)

        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 5e-4, 'momentum': 0.9},
            {'epoch': 60, 'lr': 2e-2},
            {'epoch': 120, 'lr': 4e-3},
            {'epoch': 160, 'lr': 8e-4}
        ]

def resnet_BFP(**kwargs):
    num_classes, depth, dataset, Q_method = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'Q_method'])
    print(num_classes, depth, dataset, Q_method)
    if dataset == 'cifar100':
        num_classes = num_classes or 100
        depth = depth or 56
        print('BFP_resnet_cifar100 created')
        if depth == 18:
            return ResNet_cifar100_BFP(num_classes=num_classes, block=BasicBlock_BFP, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_cifar100_BFP(num_classes=num_classes, block=BasicBlock_BFP, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_cifar100_BFP(num_classes=num_classes, block=Bottleneck_BFP, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_cifar100_BFP(num_classes=num_classes, block=Bottleneck_BFP, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_cifar100_BFP(num_classes=num_classes, block=Bottleneck_BFP, layers=[3, 8, 36, 3])
    elif dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet_BFP(num_classes=num_classes,
                                block=BasicBlock_BFP, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet_BFP(num_classes=num_classes,
                                block=BasicBlock_BFP, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet_BFP(num_classes=num_classes,
                                block=Bottleneck_BFP, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet_BFP(num_classes=num_classes,
                                block=Bottleneck_BFP, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet_BFP(num_classes=num_classes,
                                block=Bottleneck_BFP, layers=[3, 8, 36, 3])