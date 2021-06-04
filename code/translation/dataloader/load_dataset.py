from collections import defaultdict

import torch

from utils import *
from .preprocess_image import preprocess_image

import os
import re
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader

# debug
from pprint import pprint

modes = ['train', 'val', 'test']

class LowLightImage(Dataset):
    def __init__(self, args, mode):
        if mode not in modes:
            raise ValueError("mode should be %s." % (' or '.join(modes)))

        self.args = args
        self.image_path = args.image_path
        self.image_dt = self.load_images(args)
        self.mode = mode
    
    def load_images(self, args):
        image_dt = preprocess_images(args)
    
    def __len__(self):
        return len(self.image_dt)

    def __getitem__(self, idx):
        img = self.image_dt[idx]

        data = {
            'img': img,
        }

        return data

def load_data(args):
    print('Loading Low Light Data')
    
    train_dataset = LowLightData(args)

