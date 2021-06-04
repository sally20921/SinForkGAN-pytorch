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
import torch.nn as nn

from .iou import bbox_iou
from .iou import wh_iou


def build_targets(pred, targets, model):
    # targets = [image, class, x, y, w, h]
    targets_num = targets.shape[0]
    tcls, tbox, indices, anchor_vector = [], [], [], []
    reject, use_all_anchors = True, True
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i, j in enumerate(model.yolo_layers):
        # get number of grid points and anchor vec for this yolo layer
        anchors = model.module.module_list[j].anchor_vector if multi_gpu else model.module_list[j].anchor_vector

        # iou of targets-anchors
        gain[2:] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        t, a = targets * gain, []
        gwh = t[:, 4:6]
        if targets_num:
            iou = wh_iou(anchors, gwh)  # iou(3,n) = wh_iou(anchor_vec(3,2), gwh(n,2))

            if use_all_anchors:
                na = anchors.shape[0]  # number of anchors
                a = torch.arange(na).view((-1, 1)).repeat([1, targets_num]).view(-1)
                t = targets.repeat([na, 1])
            else:  # use best anchor only
                iou, a = iou.max(0)  # best iou and anchor

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            if reject:
                j = iou.view(-1) > model.hyper_parameters["iou_t"]  # iou threshold hyperparameter
                t, a = t[j], a[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4]  # grid x, y
        gwh = t[:, 4:6]  # grid w, h
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # Box
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        anchor_vector.append(anchors[a])

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, f"Model accepts {model.nc} classes labeled from 0-{model.nc - 1}, however you labelled a class {c.max()}. "

    return tcls, tbox, indices, anchor_vector


def compute_loss(pred, targets, model):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vector = build_targets(pred, targets, model)
    hyper_parameters = model.hyper_parameters  # hyperparameters
    reduction = "mean"  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([hyper_parameters["cls_pw"]]), reduction=reduction)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([hyper_parameters["obj_pw"]]), reduction=reduction)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss
    g = hyper_parameters["fl_gamma"]  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Compute losses
    np, ng = 0, 0  # number grid points, targets
    tobj = 0.
    for i, pi in enumerate(pred):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj
        np += tobj.numel()

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ng += nb
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchor_vector[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            lbox += (1.0 - giou).sum() if reduction == "sum" else (1.0 - giou).mean()  # giou loss
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn)  # targets
                t[range(nb), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    lbox *= hyper_parameters["giou"]
    lobj *= hyper_parameters["obj"]
    lcls *= hyper_parameters["cls"]
    if reduction == "sum":
        batch_size = tobj.shape[0]  # batch size
        lobj *= 3 / (6300 * batch_size) * 2  # 3 / np * 2
        if ng:
            lcls *= 3 / ng / model.nc
            lbox *= 3 / ng

    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(),
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, inputs, targets):
        loss = self.loss_fcn(inputs, targets)

        # Source: https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(inputs)  # prob from logits
        p_t = targets * pred_prob + (1 - targets) * (1 - pred_prob)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.0, 0.01, 0.99, 0.00]  # weights for [P, R, mAP, F1]@0.5 or [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
