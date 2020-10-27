import sys, os
sys.path.append(os.getcwd() + '/models')
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .Unet_phase1 import Unet_phase1, ConvBlock
torch.set_default_dtype(torch.float32)

class Unet_phase2(BaseModel):
    def __init__(self, in_channels, out_channels, is_bn=False, pretrained=''):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.activation = nn.Sigmoid()
        
        self.conv1 = nn.Conv3d(32, 4, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv3d(64, 8, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv3d(128, 16, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv3d(256, 32, kernel_size=5, stride=1, padding=2)
        
        self.up5 = nn.ConvTranspose3d(512, 96, kernel_size=2, stride=2, padding=0)
        self.conv5_de1 = ConvBlock(128, 128, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
        self.conv5_de2 = ConvBlock(128, 128, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

        self.up4 = nn.ConvTranspose3d(128, 48, kernel_size=2, stride=2, padding=0)
        self.conv4_de1 = ConvBlock(64, 64, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
        self.conv4_de2 = ConvBlock(64, 64, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

        self.up3 = nn.ConvTranspose3d(64, 24, kernel_size=2, stride=2, padding=0)
        self.conv3_de1 = ConvBlock(32, 32, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
        self.conv3_de2 = ConvBlock(32, 32, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

        self.up2 = nn.ConvTranspose3d(32, 12, kernel_size=2, stride=2, padding=0)
        self.conv2_de1 = ConvBlock(16, 16, kernel_size=3, stride=1, rate=1, is_bn=is_bn)
        self.conv2_de2 = ConvBlock(16, 16, kernel_size=3, stride=1, rate=1, is_bn=is_bn)

        self.conv1x1 = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0)

        self._initialize_weights() 

        unet_phase1 = Unet_phase1(in_channels, out_channels, is_bn)
        # Load pretrained model 
        if len(pretrained) > 0:
            checkpoint = torch.load(pretrained)
            unet_phase1.load_state_dict(checkpoint['state_dict'])
            unet_phase1 = self._freeze_model(unet_phase1)
            print("Load PreTrained Model completed!")

        base_n_filter = 16
        self.block1 = unet_phase1.block1
        
        self.block21 = unet_phase1.block21
        self.block22 = unet_phase1.block22
        
        self.block31 = unet_phase1.block31
        self.block32 = unet_phase1.block32
        
        self.block41 = unet_phase1.block41
        self.block42 = unet_phase1.block42
        
        self.block51 = unet_phase1.block51
        self.block52 = unet_phase1.block52
        del unet_phase1

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block22(self.block21(self.maxpool(x1)))
        x3 = self.block32(self.block31(self.maxpool(x2)))
        x4 = self.block42(self.block41(self.maxpool(x3)))
        x5 = self.block52(self.block51(self.maxpool(x4)))

        ux5 = self.up5(x5)
        sx4 = self.conv4(x4)
        ux5 = self.conv5_de2(self.conv5_de1(torch.cat((ux5,sx4), 1)))
        
        ux4 = self.up4(ux5)
        sx3 = self.conv3(x3)
        ux4 = self.conv4_de2(self.conv4_de1(torch.cat((ux4,sx3), 1)))
        
        ux3 = self.up3(ux4)
        sx2 = self.conv2(x2)
        ux3 = self.conv3_de2(self.conv3_de1(torch.cat((ux3,sx2), 1)))

        ux2 = self.up2(ux3)
        sx1 = self.conv1(x1)
        ux2 = self.conv2_de2(self.conv2_de1(torch.cat((ux2,sx1), 1)))

        return self.activation(self.conv1x1(ux2))
