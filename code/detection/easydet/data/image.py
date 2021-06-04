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
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from easydet.data import exif_size
from easydet.utils import xywh2xyxy
from easydet.utils import xyxy2xywh
from .common import create_folder
from .common import random_affine
from .pad_resize import letterbox

help_url = "https://github.com/Lornatang/YOLOv3-PyTorch#train-on-custom-dataset"
image_formats = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".dng"]
video_formats = [".mov", ".avi", ".mp4"]


class LoadImages:
    """ Use only in the inference phase

    Load the pictures in the directory and convert them to the corresponding format.

    Args:
        dataroot (str): The source path of the dataset.
        image_size (int): Size of loaded pictures. (default:``416``).

    """

    def __init__(self, dataroot, image_size=416):

        path = str(Path(dataroot))
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.*")))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in image_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in video_formats]
        image_num, video_num = len(images), len(videos)

        self.image_size = image_size
        self.files = images + videos
        self.files_num = image_num + video_num
        self.video_flag = [False] * image_num + [True] * video_num
        self.mode = "images"
        if any(videos):
            self.new_video(videos[0])
        else:
            self.capture = None
        assert self.files_num > 0, "No images or videos found in `" + path + "`"

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.files_num:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, raw_image = self.capture.read()
            if not ret_val:
                self.count += 1
                self.capture.release()
                # last video
                if self.count == self.files_num:
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, raw_image = self.capture.read()

            self.frame += 1
            print(f"video {self.count + 1}/{self.files_num}"
                  f"({self.frame}/{self.frames_num}) {path}: ", end="")

        else:
            # Read image
            self.count += 1
            raw_image = cv2.imread(path)  # opencv read image default is BGR
            assert raw_image is not None, "Image Not Found `" + path + "`"
            print(f"image {self.count}/{self.files_num} {path}: ", end="")

        # Padded resize operation
        image = letterbox(raw_image, new_shape=self.image_size)[0]

        # BGR convert to RGB (3 x 416 x 416)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        # Return a contiguous array
        image = np.ascontiguousarray(image)

        return path, image, raw_image, self.capture

    def new_video(self, path):
        self.frame = 0
        self.capture = cv2.VideoCapture(path)
        self.frames_num = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.files_num


class LoadImagesAndLabels(Dataset):
    """ Use in training and testing

    Load pictures and labels from the dataset and convert them to the corresponding format.

    Args:
        dataset (str): Dataset from which to load the data.
        image_size (int, optional): Size of loaded pictures. (default:``416``).
        batch_size (int, optional): How many samples per batch to load. (default: ``16``).
        augment (bool, optional): Whether image enhancement technology is needed. (default: ``False``).
        hyper_parameters (dict, optional): List of super parameters. (default: ``None``).
        rect (bool, optional): Whether to adjust to matrix training. (default: ``False``).
        image_weights (bool, optional): None. (default:``False``).
        cache_labels (bool, optional): Cache images into memory for faster training
            (WARNING: large dataset may exceed system RAM).(default:``False``).
        cache_images(bool, optional): # cache labels into memory for faster training.
            (WARNING: large dataset may exceed system RAM).(default:``False``).
        single_cls(bool, optional):  Force dataset into single-class mode. (default:``False``).
    """

    def __init__(self, dataset, image_size=416, batch_size=16, augment=False, hyper_parameters=None, rect=False,
                 image_weights=False, cache_labels=True, cache_images=False, single_cls=False):

        path = str(Path(dataset))
        assert os.path.isfile(path), f"File not found {path}. See {help_url}"
        with open(path, "r") as f:
            self.image_files = [x.replace("/", os.sep) for x in f.read().splitlines()  # os-agnostic
                                if os.path.splitext(x)[-1].lower() in image_formats]

        image_files_num = len(self.image_files)
        assert image_files_num > 0, f"No images found in {path}. See {help_url}"
        batch_index = np.floor(np.arange(image_files_num) / batch_size).astype(np.int)
        batch_num = batch_index[-1] + 1

        self.image_files_num = image_files_num
        self.batch = batch_index  # batch index of image
        self.image_size = image_size
        self.augment = augment
        self.hyper_parameters = hyper_parameters
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        # load 4 images at a time into a mosaic (only during training)
        self.mosaic = self.augment and not self.rect

        # Define labels
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.image_files]

        if self.rect:
            sp = path.replace(".txt", ".shapes")  # shapefile path
            try:
                with open(sp, "r") as f:  # read existing shapefile
                    s = [x.split() for x in f.read().splitlines()]
                    assert len(s) == self.image_files_num, "Shapefile out of sync"
            except:
                s = [exif_size(Image.open(f)) for f in
                     tqdm(self.image_files, desc="Reading image shapes")]
                np.savetxt(sp, s, fmt="%g")  # overwrites existing (if any)

            # Sort by aspect ratio
            s = np.array(s, dtype=np.float64)
            aspect_ratio = s[:, 1] / s[:, 0]  # aspect ratio
            i = aspect_ratio.argsort()
            self.image_files = [self.image_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            self.shapes = s[i]  # wh
            aspect_ratio = aspect_ratio[i]

            # Set training image shapes
            shapes = [[1, 1]] * batch_num
            for i in range(batch_num):
                ari = aspect_ratio[batch_index == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * image_size / 64.).astype(np.int) * 64

        # Preload labels (required for weighted CE training)
        self.images = [None] * self.image_files_num
        self.labels = [None] * self.image_files_num
        if cache_labels or image_weights:
            self.labels = [np.zeros((0, 5))] * self.image_files_num
            extract_bounding_boxes = False
            create_datasubset = False
            process_bar = tqdm(self.label_files, desc="Caching labels")
            nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
            file = None
            for i, file in enumerate(process_bar):
                try:
                    with open(file, "r") as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1
                    # print("missing labels for image %s" % self.image_files[i])  # file missing
                    continue

                if l.shape[0]:
                    assert l.shape[1] == 5, "> 5 label columns: %s" % file
                    assert (l >= 0).all(), "negative labels: %s" % file
                    assert (l[:,
                            1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % file
                    if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                        nd += 1  # print("WARNING: duplicate rows in %s" % self.label_files[i])  # duplicate rows
                    if single_cls:
                        l[:, 0] = 0
                    self.labels[i] = l
                    nf += 1  # file found

                    # Create subdataset (a smaller dataset)
                    if create_datasubset and ns < 1E4:
                        if ns == 0:
                            create_folder(path="./datasubset")
                            os.makedirs("./datasubset/images")
                        exclude_classes = 43
                        if exclude_classes not in l[:, 0]:
                            ns += 1
                            # shutil.copy(src=self.img_files[i], dst="./datasubset/images/")  # copy image
                            with open("./datasubset/images.txt", "a") as f:
                                f.write(self.image_files[i] + "\n")

                    # Extract object detection boxes for a second stage classifier
                    if extract_bounding_boxes:
                        p = Path(self.image_files[i])
                        img = cv2.imread(str(p))
                        h, w = img.shape[:2]
                        for j, x in enumerate(l):
                            f = "%s%sclassifier%s%g_%g_%s" % (
                                p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                            if not os.path.exists(Path(f).parent):
                                os.makedirs(Path(f).parent)  # make new output folder

                            b = x[1:] * [w, h, w, h]  # box
                            b[2:] = b[2:].max()  # rectangle to square
                            b[2:] = b[2:] * 1.3 + 30  # pad
                            b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                            b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                            b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                            assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), "Failure extracting classifier boxes"
                else:
                    # print("empty labels for image %s" % self.image_files[i])
                    # empty file
                    ne += 1
                process_bar.desc = f"Caching labels ({nf} found, {nm} missing, {ne} empty, "
                process_bar.desc += f"{nd} duplicate, for {self.image_files_num} images)"
            assert nf > 0, f"No labels found in {os.path.dirname(file) + os.sep}. See {help_url}"

        if cache_images:  # if training
            memory = 0  # Gigabytes of cached images
            process_bar = tqdm(range(len(self.image_files)), desc="Caching images")
            self.img_hw0, self.img_hw = [None] * self.image_files_num, [None] * self.image_files_num
            for i in process_bar:  # max 10k images
                # image, hw_original, hw_resized
                self.images[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)
                memory += self.images[i].nbytes
                process_bar.desc = "Caching images (%.1fGB)" % (memory / 1E9)

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io
            for file in tqdm(self.image_files, desc="Detecting corrupted images..."):
                try:
                    _ = io.imread(file)
                except IOError:
                    print(f"Corrupted image detected: {file}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        parameters = self.hyper_parameters
        if self.mosaic:
            # Load mosaic
            images, labels = load_mosaic(self, index)
            shapes = None

        else:
            # Load image
            images, (raw_height, raw_width), (height, width) = load_image(self, index)

            # Letterbox
            # final letterboxed shape
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.image_size
            images, ratio, pad = letterbox(images, shape, auto=False, scaleup=self.augment)
            # for COCO mAP rescaling
            shapes = (raw_height, raw_width), ((height / raw_height, width / raw_width), pad)

            # Load labels
            labels = []
            x = self.labels[index]
            if x is not None and x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * width * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * height * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * width * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * height * (x[:, 2] + x[:, 4] / 2) + pad[1]

        labels_num = len(labels)  # number of labels
        if labels_num:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= images.shape[0]  # height
            labels[:, [1, 3]] /= images.shape[1]  # width

        labels_out = torch.zeros((labels_num, 6))
        if labels_num:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        images = images[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        images = np.ascontiguousarray(images)

        return torch.from_numpy(images), labels_out, self.image_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        image, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(image, 0), torch.cat(label, 0), path, shapes


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    image = self.images[index]
    if image is None:  # not cached
        image_path = self.image_files[index]
        image = cv2.imread(image_path)  # BGR
        assert image is not None, "Image Not Found " + image_path
        raw_height, raw_width = image.shape[:2]  # orig hw
        r = self.image_size / max(raw_height, raw_width)  # resize image to img_size
        # always resize down, only resize up if training with augmentation
        if r < 1 or (self.augment and r != 1):
            # LINEAR for training, AREA for testing
            interpolation = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            image = cv2.resize(image,
                               (int(raw_width * r), int(raw_height * r)),
                               interpolation=interpolation)
        return image, (raw_height, raw_width), image.shape[:2]  # img, hw_original, hw_resized
    else:
        # image, hw_original, hw_resized
        return self.images[index], self.img_hw0[index], self.img_hw[index]


def load_mosaic(self, index):
    # loads images in a mosaic
    labels4 = []
    image_size = self.image_size
    # mosaic center x, y
    center_x, center_y = [int(random.uniform(image_size * 0.5, image_size * 1.5)) for _ in
                          range(2)]
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in
                         range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        image, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            # base image with 4 tiles
            image4 = np.full((image_size * 2, image_size * 2, 3), 114, dtype=np.uint8)
            # xmin, ymin, xmax, ymax (large image)
            x1a, y1a, x2a, y2a = max(center_x - w, 0), max(center_y - h, 0), center_x, center_y
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (
                    y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = center_x, max(center_y - h, 0), min(center_x + w,
                                                                     image_size * 2), center_y
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(center_x - w, 0), center_y, center_x, min(image_size * 2,
                                                                               center_y + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(center_x, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = center_x, center_y, min(center_x + w, image_size * 2), min(
                image_size * 2,
                center_y + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # image4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Load labels
        label_path = self.label_files[index]
        if os.path.isfile(label_path):
            x = self.labels[index]
            if x is None:  # labels not preloaded
                with open(label_path, "r") as f:
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            else:
                labels = np.zeros((0, 5), dtype=np.float32)
            labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * image_size, out=labels4[:, 1:])  # use with random_affine

    # Augment
    image4, labels4 = random_affine(image4, labels4,
                                    degrees=self.hyper_parameters["degrees"] * 1,
                                    translate=self.hyper_parameters["translate"] * 1,
                                    scale=self.hyper_parameters["scale"] * 1,
                                    shear=self.hyper_parameters["shear"] * 1,
                                    border=-image_size // 2)  # border to remove

    return image4, labels4


def scale_image(image, ratio=1.0, same_shape=True):  # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    height, width = image.shape[2:]
    size = (int(height * ratio), int(width * ratio))  # new size
    image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        gs = 64  # (pixels) grid size
        height, width = [math.ceil(x * ratio / gs) * gs for x in (height, width)]
    return F.pad(image, [0, width - size[1], 0, height - size[0]], value=0.447)  # value = imagenet mean


def augment_hsv(image, hgain=0.5, sgain=0.5, vgain=0.5):
    x = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    image_hsv = (cv2.cvtColor(image, cv2.COLOR_BGR2HSV) * x).clip(None, 255).astype(np.uint8)
    np.clip(image_hsv[:, :, 0], None, 179, out=image_hsv[:, :, 0])  # inplace hue clip (0 - 179 deg)
    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed
