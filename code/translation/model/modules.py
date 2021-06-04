import math
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class ConvBlock(nn.Module):
    def __init___(self, input_size, output_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, 3, 2, 1)
        self.bn = nn.InstanceNorm2d(output_size)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, 3, 2, 1, 1)
        self.bn = nn.InstanceNorm2d(output_size)
        self.relu = torch.nn.ReLU(True)

    def forward(self, x):
        out = self.deconv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out 

class ResnetBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResnetBlock, self).__init__()
        conv1 = torch.nn.Conv2d(num_filter, num_filter, 3, 1, 0)
        conv2 = torch.nn.Conv2d(num_filter, num_filter, 3, 1, 0)
        bn = torch.nn.InstanceNorm2d(num_filter)
        relu = torch.nn.ReLU(True)
        pad = torch.nn.ReflectionPad2d(1)

        self.resnet_block = torch.nn.Sequential(
                pad,
                conv1,
                bn,
                relu,
                pad, 
                conv2,
                bn
        )

    def forward(self, x):
        out = self.resnet_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, num_filter = 32):
        super(Encoder, self).__init__()
        self.pad = torch.nn.ReflectionPad2d(3)

        self.conv1 = ConvBlock(3, num_filter)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter *2, num_filter * 4)

    
    def forward(self, x):
        enc1 = self.pad(x)
        enc1 = self.conv1(enc1)
        enc2 = self.conv2(enc1)
        out = self.conv3(enc2)
        return out


class Generator(nn.Module):
    def __init__(self, num_filter = 32, num_resnet = 6):
        super(Generator, self).__init__()

        self.pad = torch.nn.ReflectionPad2d(3)

        # encoder
        #self.conv1 = ConvBlock(3, num_filter)
        #self.conv2 = ConvBlock(num_filter, num_filter * 2)
        #self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)

        # resnet blocks 
        self.resnet_blocks = []
        for i in range(num_resnet):
            self.resnet_blocks.append(ResnetBlock(num_filter * 4))
        self.resnet_blocks = torch.nn.Sequential(*self.resnet_blocks)

        # decoder
        self.deconv1 = DecovBlock(num_filter * 4, num_filter * 2)
        self.deconv2 = DeconvBlock(num_filter * 2, num_filter)
        self.deconv3 = ConvBlock(num_filter, 3)

    def forward(self, x):
        # encoder
        #enc1 = self.pad(x)
        #enc1 = self.conv1(enc1)
        #enc2 = self.conv2(enc1)
        #enc3 = self.conv3(enc2)

        # resnet blocks
        res =  self.resnset_blocks(x)

        # decoder
        dec1 = self.deconv1(res)
        dec2 = self.deconv2(dec1)
        dec2 = self.pad(dec2)
        out = self.deconv3(dec2)
        return out 


class Discriminator(nn.Module):
    def __init__(self, num_filter = 64): 
        super(Discriminator, self).__init__()

        conv1 = ConvBlock(3, num_filter)
        conv2 = ConvBlock(num_filter, num_filter * 2)
        conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        conv5 = ConvBlock(num_filter * 8, 1)

        self.conv_blocks = torch.nn.Sequential(
                conv1, 
                conv2, 
                conv3,
                conv4,
                conv5
        )

    def forward(self, x):
        out = self.conv_blocks(x)
        return out


