import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .model_utils import ConvBlock
torch.set_default_dtype(torch.float32)

class EncoderBlock(BaseModel):
    def __init__(self, in_channels, depths=3, bn_check=False, drop_rate=0.5):
        super().__init__()
        self.module = nn.ModuleDict()
        self.depth = depths

        for d in range(depths):
            fm = 2**(d + 1) * 32
            _convblock = ConvBlock(in_channels, fm, mode='3d', 
                                bn_check=bn_check, drop_rate=drop_rate, n_block=2, 
                                kernel_size=3, stride=1, padding=1)
            self.module[f'CONV_{d}'] = _convblock
            in_channels = fm
            if d != depths-1: # MaxPooling
                maxpooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
                self.module[f'MAXPOOL_{d}'] = maxpooling

    def forward(self, x):
        downsampling = [] # store to crop and copy in upsampling
        for k, op in self.module.items():
            if k.startswith('CONV_'):
                x = op(x)
                if int(k[-1]) != (self.depth-1): # prevent block center
                    downsampling.append(x)
            else:
                x = op(x)
            # print(k, x.shape)
        return x, downsampling

class DecoderBlock(BaseModel):
    def __init__(self, out_channels, depths=3, bn_check=False, drop_rate=0.5):
        super().__init__()
        self.module = nn.ModuleDict()
        self.depth = depths-1

        for d in range(self.depth-1,-1,-1):
            in_channels = 2**(d+2) *32
            transpose3d = nn.ConvTranspose3d(in_channels=in_channels, out_channels=int(in_channels/2), kernel_size=2, stride=2, padding=0)
            self.module['UPSAMPLING_{}'.format(d)] = transpose3d

            fm = int(in_channels/2)
            conv_block = ConvBlock(in_channels, fm, mode='3d', 
                                    bn_check=bn_check, drop_rate=drop_rate, n_block=2, 
                                    kernel_size=3, stride=1, padding=1)
            self.module['CONV_{}'.format(d)] = conv_block

        self.module['FINAL'] = nn.Conv3d(fm, out_channels, kernel_size=1, stride=1, padding=0)

    def crop(self, downsampling, target):
        '''
            downsampling : (b, c, d, h, w)
            target = (b, c, d, h, w)
        '''
        size_d = downsampling.shape[2:]
        size_t = target.shape[2:]
        start_d = (size_d[0] - size_t[0]) // 2
        start_h = (size_d[1] - size_t[1]) // 2
        return downsampling[:, :, start_d:(start_d + size_t[0]), start_h:(start_h + size_t[1]), start_h:(start_h + size_t[1])]

    def forward(self, x):
        x, downsampling = x[0], x[1]
        for k, op in self.module.items():
            if k.startswith('UPSAMPLING'):
                conv_i = int(k[-1])
                x = op(x)
                if x.shape != downsampling[conv_i].shape:
                    downsampling[conv_i] = self.crop(downsampling[conv_i], x)
                x = torch.cat((x, downsampling[conv_i]), dim=1)
            else:
                x = op(x)
            # print(k, x.shape)
        return x

class Unet3D(BaseModel):
    def __init__(self, in_channels, out_channels, depth, bn_check=False, drop_rate=0.0):
        super().__init__()
        self.module = nn.ModuleDict()
        self.module['ENCODER'] = EncoderBlock(in_channels, depth, bn_check, drop_rate=drop_rate)
        self.module['DECODER'] = DecoderBlock(out_channels, depth, bn_check, drop_rate=0.4)

        if out_channels==1:
            self.final = nn.Sigmoid()
        else:
            self.final = nn.Softmax(dim=1)

        self._initialize_weights()

    def forward(self, x):
        for k, op in self.module.items():
            if k.startswith('ENCODER'):
                x = op(x)
            elif k.startswith('DECODER'):
                x = op(x)
        return self.final(x)
