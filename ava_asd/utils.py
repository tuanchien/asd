# Some reusable functions used in the scripts.

# Author: Tuan Chien, James Diprose

import os
from typing import List

import click
import pandas as pd
import tensorflow
from PIL import Image


class WeightsType(click.ParamType):
    name = "weights"

    def __init__(self, *choices):
        super(WeightsType, self).__init__()
        self.choices = choices

    def convert(self, value, param, ctx):
        if value not in self.choices and not os.path.exists(value):
            msg = f'expected a value with one of the following choices: {self.choices} or a valid file path. ' \
                  f'Instead got {value}'
            self.fail(msg, param, ctx)
        return value


def create_dir(d):
    """
    Create a directory if it does not exist.
    """
    if not os.path.exists(d):
        os.makedirs(d)


def get_start_ts(extracted_path, vid_id, start_ts_filename):
    """
    Find the timestamp for the first video frame of vid_id.
    """

    ts_file = os.path.join(extracted_path, vid_id, start_ts_filename)
    with open(ts_file, 'r') as f:
        line = f.readline()
        return float(line)


def get_frame_filename(extracted_path, ann, start_ts, fps):
    """
    Get the corresponding jpg for the annotation specified.
    """

    prefix = os.path.join(extracted_path, ann.vid_id)
    frame_num = int((ann.timestamp - start_ts) * fps)
    filename = str(frame_num) + '.jpg'

    return os.path.join(prefix, filename)


def get_vid_width_height(extracted_path, vid_id):
    """
    Get the width and height of a video.
    """
    frame_path = os.path.join(extracted_path, vid_id, '1.jpg')
    width, height = Image.open(frame_path).size

    return width, height


def get_window_sizes(mode, vid_frame_size, mfcc_frame_size, sequence_size):
    """
    Calculate the correct video and audio window size for the model.
    """

    if mode == 'r_av':
        return vid_frame_size, mfcc_frame_size

    return vid_frame_size * sequence_size, mfcc_frame_size * sequence_size


def save_csv(data, columns: List, path: str, header=True):
    results_df = pd.DataFrame(data=data, columns=columns)
    results_df.dropna(inplace=True)
    results_df.to_csv(path, index=False, header=header)


def get_avr_mode(mode):
    audio = True
    video = True

    if mode == 's_av':
        audio = True
        video = True
    elif mode == 's_av_fc':
        audio = True
        video = True
    elif mode == 's_v':
        audio = False
        video = False
    elif mode == 's_a':
        audio = True
        video = False

    return audio, video


def set_gpu_memory_growth(on: bool):
    """
    Set GPU memory growth
    """

    devices = tensorflow.config.list_physical_devices('GPU')
    for device in devices:
        tensorflow.config.experimental.set_memory_growth(device, on)
