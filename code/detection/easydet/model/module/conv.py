# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activition import HSigmoid
from .activition import Mish
from .activition import Swish


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 batch_norm=None, activation=None):
        super(BasicConv2d, self).__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride, padding, bias=not batch_norm)]

        if batch_norm:
            layers.append(
                nn.BatchNorm2d(out_channels, momentum=0.003, eps=0.0001))

        if activation == "leakyrelu":
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "swish":
            layers.append(Swish())
        elif activation == "mish":
            layers.append(Mish())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)

        return out


class SeModule(nn.Module):
    """See the paper "Inverted Residuals and Linear Bottlenecks:
       Mobile Networks for Classification, Detection and Segmentation" for more details.
    """

    def __init__(self, in_channels, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            HSigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class DeepConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(DeepConv2d, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels, momentum=0.003, eps=0.0001),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.003, eps=0.0001),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.main(x)

        return out


# MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
class MixConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 5, 7), stride=1,
                 dilation=1, bias=True, method="equal_params"):
        super(MixConv2d, self).__init__()

        groups = len(kernel_size)
        if method == "equal_ch":  # equal channels per group
            i = torch.linspace(0, groups - 1E-6, out_channels).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # "equal_params": equal parameter count per group
            b = [out_channels] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(kernel_size) ** 2
            a[0] = 1
            # solve for equal weight indices, ax = b
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)

        self.m = nn.ModuleList([torch.nn.Conv2d(in_channels=in_channels,
                                                out_channels=ch[g],
                                                kernel_size=kernel_size[g],
                                                stride=stride,
                                                padding=(kernel_size[g] - 1) // 2,  # "same" pad
                                                dilation=dilation,
                                                bias=bias) for g in range(groups)])

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)
