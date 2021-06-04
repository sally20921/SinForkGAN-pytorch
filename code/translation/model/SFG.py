import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt 

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from .modules import Encoder, Generator, Discriminator

class SFG(nn.Module):
    def __init__(self, args):
        super().__init__()
        print("Model Name: SinForkGAN")
        E_x = Encoder()
        E_y = Encoder()
        G_r_x = Generator()
        G_t_y = Generator()
        G_r_y = Generator()
        G_t_x = Generator()

        

