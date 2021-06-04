import math
import os
from collections import defaultdict

import PIL
from PIL import Image
from tqdm import tqdm 

import torch
from torch import nn, utils
from torchvision import models, datasets, transforms
from utils import *

def preprocess_images(args):
    print('Preprocessing Images')

    image_path = args.image_path
    res = []
    image_lists = sorted(list(image_path.glob('*')))
    for image in image_lists:
        real_img = Image.open(image)
        real = (np.asarray(real_img) / 255.0)
        real = torch.from_numpy(real).float()
        real = real.permute(2, 0, 1) # expects .png
        res.append(real)

    return res



   
