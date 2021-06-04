from contextlib import contextmanager
from datetime import datetime
import os
import sys
import json
import pickle
import re

import six
import numpy as np
import torch

def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path, **kwargs):
    with open(path, 'w') as f:
        json.dump(data, f, **kwargs)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def get_dirname_from_args(args):
    dirname = ''
    for key in sorted(log_keys):
        dirname += '_'
        dirname += key
        dirname += '_'
        dirname += str(args[key])

    return dirname[1:]

def get_now():
    now = datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S')

def prepare_batch(args, batch):
    image_input = batch['img']
    return image_input

def wait_for_key(key="y"):
    text = ""
    while (text != key):
        text = six.moves.input("Press {} to quit: ".format(key))
        if text == key:
            print("terminating process")
        else:
            print("key {} unrecognizable".format(key))

@contextmanager
def suppress_stdout(do=True):
    if do:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
