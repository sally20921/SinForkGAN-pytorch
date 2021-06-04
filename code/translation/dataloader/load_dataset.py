from collections import defaultdict

import torch

from utils import *
from .preprocess_image import preprocess_images

import os
import re
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader

# debug
from pprint import pprint

modes = ['train', 'val', 'test']

class LowLightData(Dataset):
    def __init__(self, args, mode):
        if mode not in modes:
            raise ValueError("mode should be %s." % (' or '.join(modes)))

        self.args = args
        self.mode = mode
        self.image_path = os.path.join(args.image_path, '/'+ str(self.mode))
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
    
    train_dataset = LowLightData(args, mode='train')
    valid_dataset = LowLightData(args, mode='val')
    test_dataset = LowLightData(args, mode='test')

    train_iter = DataLoader(
            train_dataset,
            batch_size=args.batch_sizes[0],
            shuffle=args.shuffle[0],
            num_workers=args.num_workers
    )

    val_iter = DataLoader(
            valid_dataset,
            batch_sizes=args.batch_sizes[1],
            shuffle=args.shuffle[1],
            num_workers=args.num_workers
    )
    test_iter = DataLoader(
            test_dataset,
            batch_size=args.batch_sizes[2],
            shuffle=args.shuffle[2],
            num_workers=args.num_workers
    )

    return {'train': train_iter, 'val': val_iter, 'test': test_iter}

def get_iterator(args):
    iters = load_data(args)
    print('Data Loading Done')
    return iters

