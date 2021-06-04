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
import warnings
from threading import Thread

import cv2
import numpy as np

from easydet.data import letterbox


class LoadCamera:
    """ Use only in the inference phase

    Load the Camera in the local and convert them to the corresponding format.

    Args:
        pipe (int): Device index of camera. (default:``0``).
        image_size (int): Size of loaded pictures. (default:``416``).
    """

    def __init__(self, pipe=0, image_size=416):
        self.image_size = image_size

        if pipe == "0":
            # local camera
            pipe = 0

        self.pipe = pipe
        self.capture = cv2.VideoCapture(pipe)  # video capture object
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.capture.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            retval, raw_image = self.capture.read()
            raw_image = cv2.flip(raw_image, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.capture.grab()
                if n % 30 == 0:  # skip frames
                    retval, raw_image = self.capture.retrieve()
                    if retval:
                        break

        assert retval, f"Camera Error `{self.pipe}`"
        image_path = "webcam.png"
        print(f"Webcam {self.count}: ", end="")

        # Padded resize operation
        image = letterbox(raw_image, new_shape=self.image_size)[0]

        # BGR convert to RGB (3 x 416 x 416)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        # Return a contiguous array
        image = np.ascontiguousarray(image)

        return image_path, image, raw_image, None

    def __len__(self):
        return 0


class LoadStreams:
    """ For reading camera or network data

    Load data types from data flow.

    Args:
        dataroot (str): Data flow file name.
        image_size (int): Image size in default data flow. (default:``416``).
    """

    def __init__(self, dataroot, image_size=416):

        self.mode = "images"
        self.image_size = image_size

        if os.path.isfile(dataroot):
            with open(dataroot, "r") as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [dataroot]

        n = len(sources)
        self.images = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f"{i + 1}/{n}: {s}... ", end="")

            capture = cv2.VideoCapture(0 if s == "0" else s)
            assert capture.isOpened(), f"Failed to open {s}"

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) % 100
            _, self.images[i] = capture.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, capture]), daemon=True)
            print(f"Success ({width}*{height} at {fps:.2f}FPS).")
            thread.start()
        print("")

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.image_size)[0].shape for x in self.images],
                     0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            warnings.warn("WARNING: Different stream shapes detected. "
                          "For optimal performance supply similarly-shaped streams.")

    def update(self, index, capture):
        # Read next stream frame in a daemon thread
        num = 0
        while capture.isOpened():
            num += 1
            # Grabs the next frame from video file or capturing device.
            capture.grab()
            # read every 4th frame
            if num == 4:
                _, self.images[index] = capture.retrieve()
                num = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        raw_image = self.images.copy()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        image = [letterbox(x, new_shape=self.image_size, auto=self.rect)[0] for x in raw_image]

        # Stack
        image = np.stack(image, 0)

        # BGR convert to RGB (batch_size 3 x 416 x 416)
        image = image[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        # Return a contiguous array
        image = np.ascontiguousarray(image)

        return self.sources, image, raw_image, None

    def __len__(self):
        return 0
