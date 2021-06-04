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
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import torch.backends.cudnn as cudnn
import torch.onnx

from easydet.data import LoadImages
from easydet.data import LoadStreams
from easydet.model import Darknet
from easydet.model import ONNX_EXPORT
from easydet.model import apply_classifier
from easydet.model import load_classifier
from easydet.utils import load_classes
from easydet.utils import load_darknet_weights
from easydet.utils import non_max_suppression
from easydet.utils import plot_one_box
from easydet.utils import scale_coords
from easydet.utils import select_device
from easydet.utils import time_synchronized


def detect(save_image=False):
    # (320, 192) or (416, 256) or (608, 352) for (height, width)
    image_size = args.image_size
    output = args.output
    source = args.source
    weights = args.weights
    view_image = args.view_image
    save_txt = args.save_txt

    camera = False
    if source == "0" or source.startswith("rtsp") or source.startswith("http") or source.endswith(".txt"):
        camera = True

    # Initialize
    device = select_device(args.device)
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(args.cfg, image_size)

    # Load weight
    if weights.endswith(".pth"):
        model.load_state_dict(torch.load(weights, map_location=device)["state_dict"])
    else:
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        # init model
        model_classifier = load_classifier(name="resnet101", classes=2)
        # load model
        model_classifier.load_state_dict(torch.load("weights/resnet101.pth", map_location=device))
        model_classifier.to(device)
        model_classifier.eval()
    else:
        model_classifier = None

    # Migrate the model to the specified device
    model.to(device)
    # set eval model mode
    model.eval()

    # Set Dataloader
    video_path, video_writer = None, None
    if camera:
        view_image = True
        cudnn.benchmark = True
        dataset = LoadStreams(source, image_size=image_size)
    else:
        save_image = True
        dataset = LoadImages(source, image_size=image_size)

    # Get names and colors
    names = load_classes(args.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in
              range(len(names))]

    # Run inference
    start_time = time.time()
    # run once
    _ = model(torch.zeros((1, 3, image_size, image_size), device=device)) if device.type != "cpu" else None
    for image_path, image, im0s, video_capture in dataset:
        image = torch.from_numpy(image).to(device)
        image = image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        predict = model(image)[0]
        t2 = time_synchronized()

        # Apply NMS
        predict = non_max_suppression(predict, args.confidence_threshold, args.iou_threshold,
                                      multi_label=False, classes=args.classes,
                                      agnostic=args.agnostic_nms)

        # Apply Classifier
        if classify:
            predict = apply_classifier(predict, model_classifier, image, im0s)

        # Process detections
        for i, detect in enumerate(predict):  # detections per image
            if camera:  # batch_size >= 1
                p, context, im0 = image_path[i], f"{i:g}: ", im0s[i]
            else:
                p, context, im0 = image_path, "", im0s

            save_path = str(Path(output) / Path(p).name)
            context += f"{image.shape[2]}*{image.shape[3]} "  # get image size
            if detect is not None and len(detect):
                # Rescale boxes from img_size to im0 size
                detect[:, :4] = scale_coords(image.shape[2:], detect[:, :4],
                                             im0.shape).round()

                # Print results
                for classes in detect[:, -1].unique():
                    # detections per class
                    number = (detect[:, -1] == classes).sum()
                    context += f"{number} {names[int(classes)]}s, "

                # Write results
                for *xyxy, confidence, classes in detect:
                    if save_txt:  # Write to file
                        with open(save_path + ".txt", "a") as files:
                            files.write(("%e " * 6 + "\n") % (
                                *xyxy, classes, confidence))

                    if save_image or view_image:  # Add bbox to image
                        label = f"{names[int(classes)]} {confidence * 100:.2f}%"
                        plot_one_box(xyxy,
                                     im0,
                                     label=label,
                                     color=colors[int(classes)])

            # Stream results
            if view_image:
                cv2.imshow("camera", im0)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration

            # Print time (inference + NMS)
            print(f"{context}Done. {t2 - t1:.3f}s")

            # Save results (image with detections)
            if save_image:
                if dataset.mode == "images":
                    cv2.imwrite(save_path, im0)
                else:
                    if video_path != save_path:  # new video
                        video_path = save_path
                        if isinstance(video_writer, cv2.VideoWriter):
                            video_writer.release()  # release previous video writer

                        fps = video_capture.get(cv2.CAP_PROP_FPS)
                        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        video_writer = cv2.VideoWriter(save_path,
                                                       cv2.VideoWriter_fourcc(
                                                           *args.fourcc), fps,
                                                       (w, h))
                    video_writer.write(im0)

    print(f"Done. ({time.time() - start_time:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="cfgs/yolov3.cfg",
                        help="Neural network profile path. (default=cfgs/yolov3.cfg)")
    parser.add_argument("--names", type=str, default="data/coco.names",
                        help="Types of objects detected. (default=data/coco.names)")
    parser.add_argument("--weights", type=str, default="weights/yolov3.pth",
                        help="Model file weight path. (default=weights/yolov3.pth")
    parser.add_argument("--source", type=str, default="data/examples",
                        help="Image input source. (default=data/examples)")
    parser.add_argument("--output", type=str, default="output",
                        help="Output result folder. (default=output)")
    parser.add_argument("--image-size", type=int, default=608,
                        help="Size of processing picture. (default=608)")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Object confidence threshold. (default=0.3)")
    parser.add_argument("--iou-threshold", type=float, default=0.6,
                        help="IOU threshold for NMS. (default=0.6)")
    parser.add_argument("--fourcc", type=str, default="mp4v",
                        help="output video codec (verify ffmpeg support). (default=mp4v)")
    parser.add_argument("--device", default="",
                        help="device id (i.e. 0 or 0,1) or cpu. (default="")")
    parser.add_argument("--view-image", action="store_true",
                        help="Display results")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save results to *.txt")
    parser.add_argument("--classes", nargs="+", type=int,
                        help="Filter by class")
    parser.add_argument("--agnostic-nms", action="store_true",
                        help="Class-agnostic NMS")
    parser.add_argument("--augment", action="store_true",
                        help="augmented inference")
    args = parser.parse_args()
    print(args)

    with torch.no_grad():
        detect()
