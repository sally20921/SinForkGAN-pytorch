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
import argparse
import glob
import math
import os
import random
import time
import warnings

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from easydet.config import parse_data_config
from easydet.data import LoadImagesAndLabels
from easydet.model import Darknet
from easydet.solver import ModelEMA
from easydet.utils import compute_loss
from easydet.utils import fitness
from easydet.utils import init_seeds
from easydet.utils import labels_to_class_weights
from easydet.utils import labels_to_image_weights
from easydet.utils import load_darknet_weights
from easydet.utils import plot_results
from easydet.utils import print_mutation
from easydet.utils import select_device
from test import evaluate

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

hyper_parameters = {"giou": 3.54,  # giou loss gain
                    "cls": 37.4,  # cls loss gain
                    "cls_pw": 1.0,  # cls BCELoss positive_weight
                    "obj": 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
                    "obj_pw": 1.0,  # obj BCELoss positive_weight
                    "iou_t": 0.20,  # iou training threshold
                    "lr0": 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
                    "lrf": 0.0005,  # final learning rate (with cos scheduler)
                    "momentum": 0.937,  # SGD momentum
                    "weight_decay": 0.000484,  # optimizer weight decay
                    "fl_gamma": 0.0,  # focal loss gamma (default is gamma=1.5)
                    "hsv_h": 0.0138,  # image HSV-Hue augmentation (fraction)
                    "hsv_s": 0.678,  # image HSV-Saturation augmentation (fraction)
                    "hsv_v": 0.36,  # image HSV-Value augmentation (fraction)
                    "degrees": 1.98 * 0,  # image rotation (+/- deg)
                    "translate": 0.05 * 0,  # image translation (+/- fraction)
                    "scale": 0.05 * 0,  # image scale (+/- gain)
                    "shear": 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt
parameter_file = glob.glob("hyp*.txt")
if parameter_file:
    print(f"Using {parameter_file[0]}")
    for keys, value in zip(hyper_parameters.keys(), np.loadtxt(parameter_file[0])):
        hyper_parameters[keys] = value


def train():
    cfg = args.cfg
    data = args.data

    epochs = args.epochs
    batch_size = args.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = args.weights

    image_size_min, image_size_max, image_size_val = args.image_size  # image sizes (min, max, test)

    # Image Sizes
    gs = 64  # (pixels) grid size
    assert math.fmod(image_size_min, gs) == 0, f"--image-size must be a {gs}-multiple"
    args.multi_scale |= image_size_min != image_size_max  # multi if different (min, max)

    if args.multi_scale:
        if image_size_min == image_size_max:
            image_size_min //= 1.5
            image_size_max //= 0.667
        grid_min, grid_max = image_size_min // gs, image_size_max // gs
        image_size_min, image_size_max = grid_min * gs, grid_max * gs
    image_size = image_size_max  # initialize with max size

    # Configure run
    init_seeds()
    dataset_dict = parse_data_config(data)
    train_path = dataset_dict["train"]
    valid_path = dataset_dict["valid"]
    num_classes = 1 if args.single_cls else int(dataset_dict["classes"])
    hyper_parameters["cls"] = num_classes / 80  # update coco-tuned parameters["cls"] to current dataset

    # Remove previous results
    for files in glob.glob("results.txt"):
        os.remove(files)

    # Initialize model
    model = Darknet(cfg).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for model_key, model_value in dict(model.named_parameters()).items():
        if ".bias" in model_key:
            pg2 += [model_value]  # biases
        elif "Conv2d.weight" in model_key:
            pg1 += [model_value]  # apply weight_decay
        else:
            pg0 += [model_value]  # all else

    optimizer = torch.optim.SGD(pg0,
                                lr=hyper_parameters["lr0"],
                                momentum=hyper_parameters["momentum"],
                                nesterov=True)
    optimizer.add_param_group({"params": pg1,
                               # add pg1 with weight_decay
                               "weight_decay": hyper_parameters["weight_decay"]})
    optimizer.add_param_group({"params": pg2})  # add pg2 with biases
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0
    context = None
    if weights.endswith(".pth"):
        state = torch.load(weights, map_location=device)
        # load model
        try:
            state["state_dict"] = {k: v for k, v in state["state_dict"].items()
                                   if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(state["state_dict"], strict=False)
        except KeyError as e:
            error_msg = f"{args.weights} is not compatible with {args.cfg}. "
            error_msg += f"Specify --weights `` or specify a --cfg "
            error_msg += f"compatible with {args.weights}. "
            raise KeyError(error_msg) from e

        # load optimizer
        if state["optimizer"] is not None:
            optimizer.load_state_dict(state["optimizer"])
            best_fitness = state["best_fitness"]

        # load results
        if state.get("training_results") is not None:
            with open("results.txt", "w") as file:
                file.write(state["training_results"])  # write results.txt

        start_epoch = state["epoch"] + 1
        del state

    elif len(weights) > 0:
        # possible weights are "*.weights", "yolov3-tiny.conv.15",  "darknet53.conv.74" etc.
        load_darknet_weights(model, weights)
    else:
        print("Pre training model weight not loaded.")

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        # skip print amp info
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    # source https://arxiv.org/pdf/1812.01187.pdf
    lr_lambda = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.last_epoch = start_epoch - 1  # see link below

    # Initialize distributed training
    if device.type != "cpu" and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend="nccl",  # "distributed backend"
                                # distributed training init method
                                init_method="tcp://127.0.0.1:18888",
                                # number of nodes for distributed training
                                world_size=1,
                                # distributed training node rank
                                rank=0)
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.yolo_layers = model.module.yolo_layers

    # Dataset
    # Apply augmentation hyper parameters (option: rectangular training)
    train_dataset = LoadImagesAndLabels(dataset=train_path,
                                        image_size=image_size,
                                        batch_size=batch_size,
                                        augment=True,
                                        hyper_parameters=hyper_parameters,
                                        rect=args.rect,
                                        cache_images=args.cache_images,
                                        single_cls=args.single_cls)
    # No apply augmentation hyper parameters and rectangular inference
    valid_dataset = LoadImagesAndLabels(dataset=valid_path,
                                        image_size=image_size,
                                        batch_size=batch_size,
                                        augment=False,
                                        hyper_parameters=hyper_parameters,
                                        rect=True,
                                        cache_images=args.cache_images,
                                        single_cls=args.single_cls)
    collate_fn = train_dataset.collate_fn
    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=not args.rect,
                                                   pin_memory=True,
                                                   collate_fn=collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   collate_fn=collate_fn)

    # Model parameters
    model.nc = num_classes  # attach number of classes to model
    model.hyper_parameters = hyper_parameters  # attach hyper parameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # attach class weights
    model.class_weights = labels_to_class_weights(train_dataset.labels, num_classes).to(device)

    # Model EMA
    ema = ModelEMA(model, decay=0.9998)

    # Start training
    batches_num = len(train_dataloader)  # number of batches
    burns = max(3 * batches_num, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(num_classes)  # mAP per class
    # "P", "R", "mAP", "F1", "val GIoU", "val Objectness", "val Classification"
    results = (0, 0, 0, 0, 0, 0, 0)
    print(f"Using {args.workers} dataloader workers.")
    print(f"Starting training for {args.epochs} epochs...")

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()

        # Update image weights (optional)
        if train_dataset.image_weights:
            # class weights
            class_weights = model.class_weights.cpu().numpy() * (1 - maps) ** 2
            image_weights = labels_to_image_weights(train_dataset.labels,
                                                    num_classes=num_classes,
                                                    class_weights=class_weights)
            # rand weighted index
            train_dataset.indices = random.choices(range(train_dataset.image_files_num),
                                                   weights=image_weights,
                                                   k=train_dataset.image_files_num)

        mean_losses = torch.zeros(4).to(device)
        print("\n")
        print(("%10s" * 8) % ("Epoch", "memory", "GIoU", "obj", "cls", "total", "targets",
                              " image_size"))
        progress_bar = tqdm(enumerate(train_dataloader), total=batches_num)
        for index, (images, targets, paths, _) in progress_bar:
            # number integrated batches (since train start)
            ni = index + batches_num * epoch
            # uint8 to float32, 0 - 255 to 0.0 - 1.0
            images = images.to(device).float() / 255.0
            targets = targets.to(device)

            # Hyper parameter Burn-in
            if ni <= burns * 2:
                # giou loss ratio (obj_loss = 1.0 or giou)
                model.gr = np.interp(ni, [0, burns * 2], [0.0, 1.0])

                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni,
                                        [0, burns],
                                        [0.1 if j == 2 else 0.0,
                                         x["initial_lr"] * lr_lambda(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, [0, burns], [0.9, hyper_parameters["momentum"]])

            # Multi-Scale training
            if args.multi_scale:
                #  adjust img_size (67% - 150%) every 1 batch
                if ni / accumulate % 1 == 0:
                    image_size = random.randrange(image_size_min, image_size_max + 1) * gs
                scale_ratio = image_size / max(images.shape[2:])
                if scale_ratio != 1:
                    # new shape (stretched to 32-multiple)
                    new_size = [math.ceil(size * scale_ratio / gs) * gs
                                for size in images.shape[2:]]
                    images = F.interpolate(images,
                                           size=new_size,
                                           mode="bilinear",
                                           align_corners=False)

            # Run model
            output = model(images)

            # Compute loss
            loss, loss_items = compute_loss(output, targets, model)
            if not torch.isfinite(loss):
                warnings.warn(f"WARNING: Non-finite loss, ending training {loss_items}")
                return results

            # Scale loss by nominal batch_size of (16 * 4 = 64)
            loss *= batch_size / 64

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print batch results
            # update mean losses
            mean_losses = (mean_losses * index + loss_items) / (index + 1)
            memory = f"{torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0:.2f}G"
            context = ("%10s" * 2 + "%10.3g" * 6) % ("%g/%g" % (epoch, args.epochs - 1),
                                                     memory, *mean_losses,
                                                     len(targets), image_size)
            progress_bar.set_description(context)

        # Update scheduler
        scheduler.step()

        # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if not args.notest or final_epoch:  # Calculate mAP
            coco = any([coco_name in data for coco_name in ["coco.data",
                                                            "coco2014.data",
                                                            "coco2017.data"]]) and model.nc == 80
            results, maps = evaluate(cfg,
                                     data,
                                     batch_size=batch_size,
                                     image_size=image_size_val,
                                     model=ema.ema,
                                     save_json=final_epoch and coco,
                                     single_cls=args.single_cls,
                                     dataloader=valid_dataloader)

        # Write epoch results
        with open("results.txt", "a") as f:
            # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
            f.write(context + "%10.3g" * 7 % results)
            f.write("\n")

        # Write Tensorboard results
        if tb_writer:
            tags = ["train/giou_loss", "train/obj_loss", "train/cls_loss",
                    "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/F1",
                    "val/giou_loss", "val/obj_loss", "val/cls_loss"]
            for x, tag in zip(list(mean_losses[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        # fitness_i = weighted combination of [P, R, mAP, F1]
        fitness_i = fitness(np.array(results).reshape(1, -1))
        if fitness_i > best_fitness:
            best_fitness = fitness_i

        # Save training results
        save = (not args.nosave) or (final_epoch and not args.evolve)
        if save:
            with open("results.txt", "r") as f:
                # Create checkpoint
                state = {"epoch": epoch,
                         "best_fitness": best_fitness,
                         "training_results": f.read(),
                         "state_dict": ema.ema.module.state_dict()
                         if hasattr(model, "module") else ema.ema.state_dict(),
                         "optimizer": None if final_epoch else optimizer.state_dict()}

        # Save last checkpoint
        torch.save(state, "weights/checkpoint.pth")

        # Save best checkpoint
        if (best_fitness == fitness_i) and not final_epoch:
            state = {"epoch": -1,
                     "best_fitness": None,
                     "training_results": None,
                     "state_dict": ema.ema.state_dict(),
                     "optimizer": None}
            torch.save(state, "weights/model_best.pth")

        # Delete checkpoint
        del state

    if not args.evolve:
        plot_results()  # save as results.png
    print(f"{epoch - start_epoch} epochs completed "
          f"in "f"{(time.time() - start_time) / 3600:.3f} hours.\n")
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300,
                        help="500200 is YOLOv3 max batches. (default: 300)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="mini-batch size (default: 16), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel"
                             "Effective batch size is batch_size * accumulate.")
    parser.add_argument("--cfg", type=str, default="cfgs/yolov3.cfg",
                        help="Neural network profile path. "
                             "(default: cfgs/yolov3.cfg)")
    parser.add_argument("--data", type=str, default="data/coco2014.data",
                        help="Path to dataset. (default: cfgs/coco2014.data)")
    parser.add_argument("--workers", default=8, type=int, metavar="N",
                        help="Number of data loading workers (default: 8)")
    parser.add_argument("--multi-scale", action="store_true",
                        help="adjust (67% - 150%) img_size every 10 batches")
    parser.add_argument("--image-size", nargs="+", type=int, default=[320, 640],
                        help="Size of processing picture. (default: [320, 640])")
    parser.add_argument("--rect", action="store_true",
                        help="rectangular training for faster training.")
    parser.add_argument("--resume", action="store_true",
                        help="resume training from checkpoint.pth")
    parser.add_argument("--nosave", action="store_true",
                        help="only save final checkpoint")
    parser.add_argument("--notest", action="store_true",
                        help="only test final epoch")
    parser.add_argument("--evolve", action="store_true",
                        help="evolve hyperparameters")
    parser.add_argument("--cache-images", action="store_true",
                        help="cache images for faster training.")
    parser.add_argument("--weights", type=str, default="",
                        help="Model file weight path. (default: ``)")
    parser.add_argument("--device", default="",
                        help="device id (i.e. 0 or 0,1 or cpu)")
    parser.add_argument("--single-cls", action="store_true",
                        help="train as single-class dataset")
    args = parser.parse_args()
    args.weights = "weights/checkpoint.pth" if args.resume else args.weights

    print(args)
    args.image_size.extend([args.image_size[-1]] * (3 - len(args.image_size)))  # extend to 3 sizes (min, max, test)

    device = select_device(args.device, apex=mixed_precision,
                           batch_size=args.batch_size)
    if device.type == "cpu":
        mixed_precision = False

    try:
        os.makedirs("weights")
    except OSError:
        pass

    tb_writer = None
    if not args.evolve:
        try:
            # Start Tensorboard with "tensorboard --logdir=runs"
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter()
            print("Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/")
        except:
            pass

        train()

    else:  # Evolve hyperparameters (optional)
        args.notest, args.nosave = True, True  # only test/save final epoch

        for _ in range(1):  # generations to evolve
            # if evolve.txt exists: select best hyps and mutate
            if os.path.exists("evolve.txt"):
                # Select parent(s)
                parent = "single"  # parent selection method: "single" or "weighted"
                x = np.loadtxt("evolve.txt", ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == "single" or len(x) == 1:
                    # weighted selection
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                np.random.seed(int(time.time()))
                # gains
                g = np.array(
                    [1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])
                ng = len(g)
                if method == 1:
                    v = (np.random.randn(ng) *
                         np.random.random() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (np.random.randn(ng) *
                         np.random.random(ng) * g * s + 1) ** 2.0
                elif method == 3:
                    v = np.ones(ng)
                    # mutate until a change occurs (prevent duplicates)
                    while all(v == 1):
                        v = (g * (np.random.random(ng) < mp) *
                             np.random.randn(ng) *
                             np.random.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(
                        hyper_parameters.keys()):  # plt.hist(v.ravel(), 300)
                    hyper_parameters[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ["lr0", "iou_t", "momentum", "weight_decay", "hsv_s",
                    "hsv_v", "translate", "scale", "fl_gamma"]
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001),
                      (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyper_parameters[k] = np.clip(hyper_parameters[k], v[0], v[1])

            # Train mutation
            res = train()

            # Write mutation results
            print_mutation(hyper_parameters, res)
