import sys, os
sys.path.append(os.getcwd() + '/models')
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
torch.set_default_dtype(torch.float32)

class ConvBlock(BaseModel):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, rate=1, is_bn=False):
        super().__init__()
        self.module = torch.nn.Sequential()    
        self.module.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=rate, dilation=rate))
        if is_bn:
            self.module.add_module('bn', nn.BatchNorm3d(out_channels))
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.module(x))
        
class Unet_phase1(BaseModel):
    def __init__(self, in_channels, out_channels, is_bn=True):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.activation = nn.Sigmoid()

        self.block1 = ConvBlock(in_channels, 16, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

        self.block21 = ConvBlock(16, 32, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
        self.block22 = ConvBlock(32, 32, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

        self.block31 = ConvBlock(32, 64, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
        self.block32 = ConvBlock(64, 64, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

        self.block41 = ConvBlock(64, 128, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
        self.block42 = ConvBlock(128, 128, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

        self.block51 = ConvBlock(128, 256, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
        self.block52 = ConvBlock(256, 256, kernel_size=3, stride=1, rate=1, is_bn=is_bn) # no maxpool

        self.up2 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0)
        self.score_aux1 = nn.Conv3d(16, out_channels, kernel_size=1, stride=1, padding=0)

        self.up31 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
        self.up32 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0)
        self.score_aux2 = nn.Conv3d(16, out_channels, kernel_size=1, stride=1, padding=0)

        self.up41 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0)
        self.up42 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
        self.up43 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0)
        self.score_aux3 = nn.Conv3d(16, out_channels, kernel_size=1, stride=1, padding=0)

        self.up51 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, padding=0)
        self.up52 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0)
        self.up53 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
        self.up54 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0)
        self.score_main = nn.Conv3d(16, out_channels, kernel_size=1, stride=1, padding=0)
        self._initialize_weights()

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block22(self.block21(self.maxpool(x1)))
        x3 = self.block32(self.block31(self.maxpool(x2)))
        x4 = self.block42(self.block41(self.maxpool(x3)))
        x5 = self.block52(self.block51(self.maxpool(x4)))

        ux1  = self.score_aux1(self.up2(x2))
        ux2  = self.score_aux2(self.up32(self.up31(x3)))
        ux3  = self.score_aux3(self.up43(self.up42(self.up41(x4))))
        main = self.score_main(self.up54(self.up53(self.up52(self.up51(x5)))))
        return self.activation(ux1), self.activation(ux2), self.activation(ux3), self.activation(main)

# import sys, os
# sys.path.append(os.getcwd() + '/models')
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from base import BaseModel
# torch.set_default_dtype(torch.float32)

# class ConvBlock(BaseModel):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, rate=1, is_bn=False):
#         super().__init__()
#         self.module = torch.nn.Sequential()    
#         self.module.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=rate, dilation=rate))
#         if is_bn:
#             self.module.add_module('bn', nn.BatchNorm3d(out_channels))
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         return self.relu(self.module(x))
        
# class Unet_phase1(BaseModel):
#     def __init__(self, in_channels, out_channels, is_bn=True):
#         super().__init__()
#         self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
#         self.activation = nn.Sigmoid()

#         self.block1 = ConvBlock(in_channels, 32, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

#         self.block21 = ConvBlock(32, 64, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
#         self.block22 = ConvBlock(64, 64, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

#         self.block31 = ConvBlock(64, 128, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
#         self.block32 = ConvBlock(128, 128, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

#         self.block41 = ConvBlock(128, 256, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
#         self.block42 = ConvBlock(256, 256, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

#         self.block51 = ConvBlock(256, 512, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
#         self.block52 = ConvBlock(512, 512, kernel_size=3, stride=1, rate=1, is_bn=is_bn) # no maxpool

#         self.up2 = nn.ConvTranspose3d(64, 16, kernel_size=2, stride=2, padding=0)
#         self.score_aux1 = nn.Conv3d(16, out_channels, kernel_size=1, stride=1, padding=0)

#         self.up31 = nn.ConvTranspose3d(128, 32, kernel_size=2, stride=2, padding=0)
#         self.up32 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0)
#         self.score_aux2 = nn.Conv3d(16, out_channels, kernel_size=1, stride=1, padding=0)

#         self.up41 = nn.ConvTranspose3d(256, 64, kernel_size=2, stride=2, padding=0)
#         self.up42 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
#         self.up43 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0)
#         self.score_aux3 = nn.Conv3d(16, out_channels, kernel_size=1, stride=1, padding=0)

#         self.up51 = nn.ConvTranspose3d(512, 128, kernel_size=2, stride=2, padding=0)
#         self.up52 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0)
#         self.up53 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
#         self.up54 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0)
#         self.score_main = nn.Conv3d(16, out_channels, kernel_size=1, stride=1, padding=0)
#         self._initialize_weights()

#     def forward(self, x):
#         x1 = self.block1(x)
#         x2 = self.block22(self.block21(self.maxpool(x1)))
#         x3 = self.block32(self.block31(self.maxpool(x2)))
#         x4 = self.block42(self.block41(self.maxpool(x3)))
#         x5 = self.block52(self.block51(self.maxpool(x4)))

#         ux1  = self.score_aux1(self.up2(x2))
#         ux2  = self.score_aux2(self.up32(self.up31(x3)))
#         ux3  = self.score_aux3(self.up43(self.up42(self.up41(x4))))
#         main = self.score_main(self.up54(self.up53(self.up52(self.up51(x5)))))
#         return self.activation(ux1), self.activation(ux2), self.activation(ux3), self.activation(main)
