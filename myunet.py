import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
     return nn.Sequential(
     nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
     nn.BatchNorm2d(out_channels),
     nn.ReLU(inplace=True),
     )

class UNet_gomoku (nn.Module):

    def __init__(self, n_class=1):
        super(UNet_gomoku, self).__init__()
        self.down1 = nn.Sequential(
            make_conv_bn_relu(2, 8, kernel_size=3, stride=1, padding=0 ),
            make_conv_bn_relu(8, 8, kernel_size=3, stride=1, padding=0 ),
        )
       

        self.down2 = nn.Sequential(
            make_conv_bn_relu(8, 16, kernel_size=3, stride=1, padding=1 ),
            make_conv_bn_relu(16,16, kernel_size=3, stride=1, padding=0 ),
        )
        

        self.same = nn.Sequential(
            make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )


        self.up2 = nn.Sequential(
            make_conv_bn_relu(32, 16, kernel_size=3, stride=1, padding=2 ),
            make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        self.up1 = nn.Sequential(
            make_conv_bn_relu(24, 8, kernel_size=1, stride=1, padding=2 ),
            make_conv_bn_relu(8, 8, kernel_size=3, stride=1, padding=2 ),
        )
        #128

        self.classify = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=0 )



    def forward(self, x):

        #128

        down1 = self.down1(x)

        down2 = self.down2(down1)

        out   = self.same(down2)

        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = self.classify(out)
        #out   = F.sigmoid(out)

        return out
    
