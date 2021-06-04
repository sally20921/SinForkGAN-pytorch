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

import glob
import random
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc("font", **{"size": 11})

# format short g, %precision=5
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})

# Prevent OpenCV from multi threading (to use PyTorch DataLoader)
# for apply_classifier and plot one box
cv2.setNumThreads(0)


def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    # line thickness
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)


def plot_results(start=0, stop=0):  # from utils.utils import *; plot_results()
    # Plot training results files "results*.txt"
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    ax = ax.ravel()
    s = ["GIoU", "Objectness", "Classification", "Precision", "Recall",
         "val GIoU", "val Objectness", "val Classification", "mAP@0.5", "F1"]

    files = glob.glob("results*.txt")

    for f in sorted(files):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            y = results[i, x]
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan  # dont show zero loss values
            ax[i].plot(x, y, marker=".", label=Path(f).stem, linewidth=2, markersize=8)
            ax[i].set_title(s[i])
            if i in [5, 6, 7]:  # share train and val loss y axes
                ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])

    fig.tight_layout()
    ax[1].legend()
    fig.savefig("result.png", dpi=200)
