# Copyright 2020 Tuan Chien, James Diprose
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Tuan Chien, James Diprose

import os
import pandas as pd
import pickle
from typing import Tuple

BoundingBox = Tuple[float, float, float, float]
FaceSize = Tuple[float, float]


class Annotation:
    """
    AVA Active Speaker Detection annotations.

    Parameters
    ----------
    vid_id : str
        Video id string
    timestamp : float
        Timestamp in seconds for this annotation.
    bbox : (x1,y1,x2,y2)
        Bounding box for the face.
    label : str
        Classification label.
    face_id : str
        Unique face id.
    face_size : (width, height)
        Size of the face in pixels.
    """

    def __init__(self, vid_id: str, timestamp: float, bbox: BoundingBox, label: str, face_id: str, face_size: FaceSize):
        self.vid_id = vid_id
        self.timestamp = timestamp
        self.bbox = bbox
        self.label = label
        self.face_id = face_id
        self.face_size = face_size


# Utility functions

def vid_id_from_filename(vid_filename):
    """
    Get the video id from the filename.
    """
    return vid_filename.split('.')[0]


def find_annotation_file(vid, train_ann_dir, test_ann_dir):
    """
    Find the annotation filename for the video file.
    """
    suffix = '-activespeaker.csv'

    vid_id = vid_id_from_filename(vid)
    vid_file = vid_id + suffix

    paths = [os.path.join(test_ann_dir, vid_file),
             os.path.join(train_ann_dir, vid_file)]

    for path in paths:
        if os.path.exists(path):
            return path

    raise Exception('Annotation not found for ' + vid)


def get_min_max_timestamp(ann_file, eps=0.0):
    """
    Get the minimum and maximum timestamp with some epsilon buffer for a given annotation file.
    """
    df = pd.read_csv(ann_file, header=None, delimiter=',')
    min_ts = float('inf')
    max_ts = 0

    for i in range(len(df)):
        timestamp = df[1][i]
        min_ts = min(min_ts, timestamp)
        max_ts = max(max_ts, timestamp)

    return min_ts - eps, max_ts + eps


def load_annotation_file(filename):
    """
    Load the annotation file.
    """
    return pickle.load(open(filename, 'rb'))
