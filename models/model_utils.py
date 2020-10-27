import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
torch.set_default_dtype(torch.float32)

class ConvBlock(BaseModel):
    def __init__(self, in_channels, out_channels, mode='3d', 
                 bn_check=False, active_final=True, drop_rate=0.0, n_block=2, 
                 kernel_size=3, stride=1, padding=1):
        '''
            mode: '2d' or '3d'
            ConvBlock comprises 
                (Conv -> Dropout -> BatchNorm (if bn_check) -> Relu (if active_final) )^n_block
        '''
        super().__init__()
        self.module = torch.nn.Sequential()
        if mode=='3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
            dropout = nn.Dropout3d
        elif mode=='2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
            dropout = nn.Dropout2d
        else:
            raise NameError("mode is not valid")

        # check whether using BatchNorm or not
        n = 4 if bn_check else 3
        for i in range(0, n_block*n, n):
            self.module.add_module(str(i), conv(in_channels, out_channels, kernel_size, stride, padding))
            self.module.add_module(str(i+1), dropout(drop_rate))
            if bn_check:
                self.module.add_module(str(i+2), bn(out_channels))
                if i == n*(n_block-1) and not active_final:
                    break
                self.module.add_module(str(i+3), nn.ReLU(inplace=True))
            else:
                if i == n*(n_block-1) and not active_final:
                    break
                self.module.add_module(str(i+2), nn.ReLU(inplace=True))
            in_channels = out_channels

    def forward(self, x):
        return self.module(x)

class ResBlock(BaseModel):
    def __init__(self, in_channels, out_channels, mode='3d',
                 use_conv1=False, bn_check=False, drop_rate=0.0):
        super().__init__()
        if mode=='3d':
            conv = nn.Conv3d
        elif mode=='2d':
            conv = nn.Conv2d
        else:
            raise NameError("mode is not valid")
        self.convblock = ConvBlock(in_channels, out_channels, mode=mode,
                                   bn_check=bn_check, active_final=False, drop_rate=drop_rate, n_block=2, 
                                   kernel_size=3, stride=1, padding=1)
        self.shortcut = conv(in_channels, out_channels, kernel_size=1) if use_conv1 else False
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.convblock(x)
        y_res = self.shortcut(x) if self.shortcut else x
        return self.activation(y + y_res)