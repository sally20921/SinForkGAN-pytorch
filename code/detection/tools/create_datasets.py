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

# Source: https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
import os
import xml.etree.ElementTree
import argparse
from PIL import Image

sets = ["train", "val", "trainval"]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(xml_path, image_index):
    in_file = open(f"{xml_path}/{image_index}.xml")
    out_file = open(f"labels/{image_index}.txt", "w")
    tree = xml.etree.ElementTree.parse(in_file)
    root = tree.getroot()

    w = 0
    h = 0
    try:
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
    except ValueError:
        pass
    else:
        path = os.path.join(os.getcwd(), "JPEGImages", image_index + ".jpg")
        img = Image.open(path)
        w, h = img.size

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        box = (float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text), float(xmlbox.find("ymin").text),
               float(xmlbox.find("ymax").text))
        bbox = convert((w, h), box)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bbox]) + "\n")


def main(args):
    try:
        os.makedirs("labels")
    except OSError:
        pass

    for image_set in sets:
        image_indexs = open(f"ImageSets/Main/{image_set}.txt").read().strip().split()
        list_file = open(f"{image_set}.txt", "w")
        for image_index in image_indexs:
            list_file.write(f"data/{args.dataroot}/images/{image_index}.jpg\n")
            convert_annotation(args.xml_path, image_index)
        list_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script tool for dividing training set and verification set in dataset.")
    parser.add_argument('--xml-path', type=str, default="./Annotations", help="Location of dimension files in dataset.")
    parser.add_argument('--dataroot', type=str, required=True, help='Dataset name')
    args = parser.parse_args()

    main(args)