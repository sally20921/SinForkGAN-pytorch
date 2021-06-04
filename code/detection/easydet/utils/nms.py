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
import torch
import torchvision

from .coords import xywh2xyxy
from .iou import box_iou


def non_max_suppression(prediction,
                        confidence_threshold=0.1,
                        iou_threshold=0.6,
                        multi_label=True,
                        classes=None,
                        agnostic=False):
    """
        Performs  Non-Maximum Suppression on inference results
        Returns detections with shape:
            nx6 (x1, y1, x2, y2, conf, cls)
        """

    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    method = "merge"
    num_classes = prediction[0].shape[1] - 5  # number of classes
    multi_label &= num_classes > 1  # multiple labels per box
    output = [None] * len(prediction)
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply conf constraint
        x = x[x[:, 4] > confidence_threshold]

        # Apply width-height constraint
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > confidence_threshold).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        if not torch.isfinite(x).all():
            x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]
        i = 0.
        if method == "merge":  # Merge NMS (boxes merged using weighted mean)
            i = torchvision.ops.boxes.nms(boxes, scores, iou_threshold)
            if n < 1e4:  # update boxes
                weights = (box_iou(boxes[i], boxes) > iou_threshold) * scores[None]  # box weights
                # merged boxes
                x[i, :4] = torch.mm(weights / weights.sum(1, keepdim=True), x[:, :4]).float()
        elif method == "vision":
            i = torchvision.ops.boxes.nms(boxes, scores, iou_threshold)
        elif method == "fast":  # FastNMS from https://github.com/dbolya/yolact
            iou = box_iou(boxes, boxes).triu_(diagonal=1)  # upper triangular iou matrix
            i = iou.max(0)[0] < iou_threshold

        output[xi] = x[i]
    return output
