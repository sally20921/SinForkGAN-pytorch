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
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def init_seeds(seed=0):
    torch.manual_seed(seed)
    # If you or any of the libraries you are using rely on Numpy,
    # you should seed the Numpy RNG as well
    np.random.seed(0)

    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False


def select_device(device="", apex=False, batch_size=None):
    # device = "cpu" or "cuda:0"
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:  # if device requested other than "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % gpu_count == 0, f"batch-size {batch_size} not multiple of GPU count {gpu_count}"
        x = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
        s = "Using CUDA " + ("Apex " if apex else "")
        for i in range(0, gpu_count):
            if i == 1:
                s = " " * len(s)
            memory = x[i].total_memory / c
            print(f"{s}\n\t+ device:{i} (name=`{x[i].name}`, total_memory={int(memory)}MB)")
    else:
        print("Using CPU")

    print("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
