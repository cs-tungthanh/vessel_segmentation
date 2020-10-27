import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
torch.set_default_dtype(torch.float32)

class ConvBlock(BaseModel):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_bn=True, is_res=True, drop_rate=0.0):
        super().__init__()
        self.is_res = is_res
        self.module = nn.Sequential()
        self.module.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))
        if is_bn:
            self.module.add_module('bn', nn.BatchNorm3d(out_channels))
        self.module.add_module('drop', nn.Dropout3d(drop_rate))
        if is_res:
            self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.is_res:
            return self.relu(self.module(x) + self.conv1x1(x))
        return self.relu(self.module(x))

class ResUBlock(BaseModel):
    def __init__(self, in_channels, out_channels, depth=3, is_bn=True, is_res=True, drop_rate=0.0):
        super().__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv_block0 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_bn=is_bn, is_res=is_res, drop_rate=drop_rate)
        self.encoder = nn.ModuleDict()
        self.mid_conv = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, is_bn=is_bn, is_res=is_res, drop_rate=drop_rate)
        self.decoder = nn.ModuleDict()
        self.last_conv = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, is_bn=is_bn, is_res=is_res, drop_rate=drop_rate)

        # U Block
        for d in range(1, depth):
            self.encoder[f'EnConv_{d}'] = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, is_bn=is_bn, is_res=is_res, drop_rate=drop_rate)
            self.encoder[f'Down_{d}'] = self.maxpool

        for d in range(depth-1, 0, -1):
            self.decoder[f'Up_{d}'] = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
            self.decoder[f'DeConv_{d}'] = ConvBlock(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, is_bn=is_bn, is_res=is_res, drop_rate=drop_rate)

    def forward(self, x):
        hx = {}
        x0 = self.conv_block0(x)
        temp = x0
        # Encoder forward
        for k, op in self.encoder.items():
            temp = op(temp)
            if k.startswith('EnConv'):
                hx[k[-1]] = temp
        # Bridge forward - mid conv
        temp = self.mid_conv(temp)

        # Decoder forward
        for k, op in self.decoder.items():
            if k.startswith('DeConv'):
                temp = op(torch.cat((temp, hx[k[-1]]), dim=1))
            else:
                temp = op(temp)
        hx = None
        return self.last_conv(temp + x0)

class U2net3D(BaseModel):
    def __init__(self, in_channels=1, out_channels=1, depth=4, start_channels=32, u_depth=3, decrease_u_depth=True, is_bn=True, is_res=True, drop_rate=0.0):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        ori_start_channels = start_channels

        # Input
        self.first_block = nn.Sequential()
        self.first_block.add_module('conv_0',
                ConvBlock(in_channels, start_channels, kernel_size=3, stride=1, padding=1, is_bn=is_bn, is_res=is_res, drop_rate=drop_rate))
        self.first_block.add_module('conv_1',
                ConvBlock(start_channels, start_channels, kernel_size=3, stride=1, padding=1, is_bn=is_bn, is_res=is_res, drop_rate=drop_rate))

        # Encoder
        self.encoder = nn.ModuleDict()
        for d in range(1, depth):
            self.encoder[f'Down_{d}'] = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
            self.encoder[f'EnResU_{d}'] = ResUBlock(start_channels, start_channels*2, u_depth, is_bn=is_bn, is_res=is_res, drop_rate=drop_rate)
            start_channels *= 2
            if decrease_u_depth:
                u_depth -= 1

        # Decoder
        self.decoder = nn.ModuleDict()
        for d in range(1, depth):
            up_start_channels = 2**d*ori_start_channels
            aux = nn.ModuleDict()
            for i in range(1, d+1):
                aux[f'{i}'] = nn.ConvTranspose3d(up_start_channels, up_start_channels//2, kernel_size=2, stride=2)
                up_start_channels = up_start_channels//2
            aux[f'conv1x1_{d}'] = nn.Conv3d(up_start_channels, out_channels, kernel_size=1)
            self.decoder[f'Aux_{d}'] = aux

        self._initialize_weights()

    def forward(self, x):
        hx = {}
        y = {}
        # Input forward
        x = self.first_block(x)
        # Encoder forward
        for k, op in self.encoder.items():
            x = op(x)
            if k.startswith('EnResU'):
                hx[k[-1]] = x

        # Decoder forward
        for k1, op1 in self.decoder.items():
            aux_key = k1[-1]
            count = int(aux_key)
            temp_y = hx[str(aux_key)]
            for k2, op2 in op1.items():
                if 'conv1x1' not in k2: # convtranspose
                    temp_y = op2(temp_y)
                    if count > 1:
                        temp_y = temp_y + hx[str(count-1)]
                    count -= 1
                else: # conv1x1
                    temp_y = self.activation( op2(temp_y) )
            y[str(aux_key)] = temp_y
        return [y[k] for k in y]
