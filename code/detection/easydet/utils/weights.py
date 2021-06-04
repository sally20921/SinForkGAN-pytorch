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
from pathlib import Path

import numpy as np
import torch
from ..model.module import Darknet


def convert(config="cfgs/yolov3.cfg", weight="weights/yolov3.weights"):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfgs/yolov3.cfg', 'weights/yolov3.weights')

    # Initialize model
    model = Darknet(config)

    # Load weights and save
    if weight.endswith(".pth"):
        model.load_state_dict(torch.load(weight, map_location="cpu")["state_dict"])
        save_weights(model, path="converted.weights", cutoff=-1)
        print(f"Success: converted `{weight}` to `converted.weights`")

    elif weight.endswith(".weights"):
        load_darknet_weights(model, weight)

        state = {"epoch": -1,
                 "best_fitness": None,
                 "training_results": None,
                 "state_dict": model.state_dict(),
                 "optimizer": None}

        torch.save(state, "converted.pth")
        print(f"Success: converted `{weight}` to `converted.pth`")

    else:
        print("Error: extension not supported.")


def convert_to_best(config="cfgs/yolov3.cfg", weight="weights/checkpoint.pth"):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfgs/yolov3.cfg', 'weights/yolov3.weights')

    # Initialize model
    model = Darknet(config)

    # Load weights and save
    if weight.endswith(".pth"):
        model.load_state_dict(torch.load(weight, map_location="cpu")["state_dict"])
        state = {"epoch": -1,
                 "best_fitness": None,
                 "training_results": None,
                 "state_dict": model.state_dict(),
                 "optimizer": None}
        torch.save(state, "model_best.pth")
        print(f"Success: converted `{weight}` to `model_best.pth`")

    else:
        print("Error: Only support PyTorch weights file.")


def labels_to_class_weights(labels, num_classes=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=num_classes)  # occurences per class

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, num_classes=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int),
                                         minlength=num_classes) for i in range(n)])
    image_weights = (class_weights.reshape(1, num_classes) * class_counts).sum(1)
    return image_weights


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in "weights"

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == "darknet53.conv.74":
        cutoff = 75
    elif file == "yolov3-tiny.conv.15":
        cutoff = 15

    # Read weights file
    with open(weights, "rb") as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32,
                                   count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64,
                                count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(
            zip(self.module_defines[:cutoff], self.module_list[:cutoff])):
        if mdef["type"] == "convolutional":
            conv = module[0]
            if mdef["batch_normalize"]:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(
                    torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(
                    torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(
                torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path="model.weights", cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, "wb") as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (module_define, module) in enumerate(
                zip(self.module_defines[:cutoff], self.module_list[:cutoff])):
            if module_define["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_define["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)
