from pathlib import Path
import random

from fire import Fire
from munch import Munch

import torch
import numpy as np

from config import config, debug_options
from dataloader.load_dataset import get_iterator
from utils import wait_for_key, suppress_stdout
#from train import train
#from evaluate import evaluate
#from infer import infer

class Cli:
    def __init__(self):
        self.defaults = config
        self.debug = debug_options


    def _default_args(self, **kwargs):
        args = self.defaults
        if 'debug' in kwargs:
            args.update(self.debug)
        args.update(kwargs)
        args.update(resolve_paths(config))
        args.update(fix_seed(args))
        args.update(get_device(args))
        print(args)

        return Munch(args)

    def check_dataloader(self, **kwargs):
        from dataloader.load_dataset import modes
        from utils import prepare_batch
        from tqdm import tqdm

        args = self._default_args(**kwargs)
        iters = get_iterator(args)
        train_iter_test = next(iter(iters['train']))
        
        for key, value in train_iter_test.items():
            if isinstance(value, torch.Tensor):
                print(key, value.shape)
            else:
                print("Somehow the image did not translate into torch Tensor")


        for mode in modes:
            print('Test loading %s data' % mode)
            for batch in tqdm(iters[mode]):
                batch = prepare_batch(args, batch)


def resolve_paths(config):
    paths = [k for k in config.keys() if k.endswith('_path')]
    res = {}
    root = Path('../../').resolve()
    for path in paths:
        res[path] = root / config[path]
        print(res[path])

    return res

def fix_seed(args):
    if 'random_seed' not in args:
        args['random_seed'] = 0
    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])
    return args

def get_device(args):
    if 'device' in args:
        device = args['device']
    else: 
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {'device': device}

if __name__ == "__main__":
    Fire(Cli)
