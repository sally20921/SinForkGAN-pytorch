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

import numpy as np

# Check all fields are supported
supported = ["type", "in_channels", "out_channels", "in_features", "out_features",
             "num_features", "batch_normalize", "filters", "size", "stride", "pad", "activation",
             "layers", "groups", "from", "mask", "anchors", "classes", "num", "jitter",
             "ignore_thresh", "truth_thresh", "random", "stride_x", "stride_y", "weights_type",
             "weights_normalization", "scale_x_y", "beta_nms", "nms_kind", "iou_loss", "padding",
             "iou_normalizer", "cls_normalizer", "iou_thresh", "expand_size", "semodules"]


def parse_model_config(path):
    # Parse the yolo *.cfgs file and return module definitions path may be "cfgs/yolov3.cfgs", "yolov3.cfgs", or "yolov3"
    if not path.endswith(".cfg"):  # add .cfgs suffix if omitted
        path += ".cfg"
    if not os.path.exists(path) and os.path.exists(
            "cfgs" + os.sep + path):  # add cfgs/ prefix if omitted
        path = "cfgs" + os.sep + path

    with open(path, "r") as f:
        lines = f.read().split("\n")
    lines = [x for x in lines if x and not x.startswith("#")]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    modules = []
    for line in lines:
        if line.startswith("["):  # This marks the start of a new block
            modules.append({})
            modules[-1]["type"] = line[1:-1].rstrip()
            if modules[-1]["type"] == "convolutional":
                # pre-populate with zeros (may be overwritten later)
                modules[-1]["batch_normalize"] = 0
        else:
            key, value = line.split("=")
            key = key.rstrip()

            if key == "anchors":  # return numpy array
                modules[-1][key] = np.array([float(x) for x in value.split(",")]).reshape(
                    (-1, 2))  # np anchors
            elif (key in ["from", "layers", "mask"]) or (key == "size" and "," in value):
                modules[-1][key] = [int(x) for x in value.split(",")]
            else:
                value = value.strip()
                if value.isnumeric():  # return int or float
                    modules[-1][key] = int(value) if (int(value) - float(value)) == 0 else float(
                        value)
                else:
                    modules[-1][key] = value  # return string

    f = []  # fields
    for x in modules[1:]:
        [f.append(k) for k in x if k not in f]
    unsupported_field_list = [x for x in f if x not in supported]
    assert not any(
        unsupported_field_list), f"Unsupported fields {unsupported_field_list} in {path}."

    return modules
