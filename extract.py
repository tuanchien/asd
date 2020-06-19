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

# A script that extracts audio-video and mfccs from the downloaded youtube videos.
#
# av:
#    Extracts .jpgs and .wav from the downloaded youtube videos in the ava dataset.
#   A new directory is created for each video to put its extracted files.
#   The script can optionally read the annotation information and prune off the unneeded video frames.
#
# mfcc:
#    Extract MFCCs from the dataset.
# anns:
#   Organise all the annotation data so that the data generator we feed into keras can just load the file output by
#   this and do its thing.
#
# Author: Tuan Chien, James Diprose

import datetime
import os
import pickle
import random
from subprocess import PIPE

import click
import librosa
import pandas as pd
from ffmpy import FFmpeg

from ava_asd.annotation import Annotation
from ava_asd.annotation import vid_id_from_filename, get_min_max_timestamp, find_annotation_file
from ava_asd.config import read_config
from ava_asd.mfcc import Mfcc
from ava_asd.utils import create_dir
from ava_asd.utils import get_vid_width_height


@click.group()
def main():
    pass


def save_start_ts(ts, extracted_path, vid_id, start_ts_filename):
    """
    Save the timestamp corresponding to 1.jpg to a file.
    """
    out_path = os.path.join(extracted_path, vid_id, start_ts_filename)
    with open(out_path, 'w') as f:
        f.write(str(ts))


def extract_av(vid, test_ann_dir, train_ann_dir, fps, eps, start_ts_filename, vid_save_path, extracted_path):
    """
    Extract audio and video frames.
    """

    vid_in = os.path.join(vid_save_path, vid)
    vid_id = vid_id_from_filename(vid)

    out_path = os.path.join(extracted_path, vid_id)
    create_dir(out_path)

    audio_out = os.path.join(out_path, 'audio.wav')
    vid_out = os.path.join(out_path, '%d.jpg')

    ann_file = find_annotation_file(vid, train_ann_dir, test_ann_dir)
    min_ts, max_ts = get_min_max_timestamp(ann_file, eps=eps)

    save_start_ts(min_ts, extracted_path, vid_id, start_ts_filename)

    start_ts = str(datetime.timedelta(seconds=min_ts))
    end_ts = str(datetime.timedelta(seconds=max_ts))
    ff = FFmpeg(inputs={vid_in: ['-ss', start_ts, '-to', end_ts]}, outputs={
        audio_out: None, vid_out: ['-filter:v', 'fps=fps=' + str(fps), '-y']})
    ff.run(stderr=PIPE)


@main.command()
@click.argument('config-file', type=click.File('r'))
@click.argument('data-path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
def videos(config_file, data_path):
    """ Extract audio and video frames.

    CONFIG_FILE: the config file with settings for the experiment.
    DATA_PATH: the path to the folder with the data files.
    """

    config = read_config(config_file.name)
    test_ann_dir = os.path.join(data_path, config['test_ann_dir'])
    train_ann_dir = os.path.join(data_path, config['train_ann_dir'])
    vid_save_path = os.path.join(data_path, config['vid_save_path'])
    extracted_path = os.path.join(data_path, config['extracted_path'])
    fps = config['fps']
    eps = config['eps']
    start_ts_filename = config['start_ts']

    if not os.path.exists(vid_save_path):
        raise Exception('video directory ' + vid_save_path + ' does not exist')

    create_dir(extracted_path)

    vids = os.listdir(vid_save_path)
    nvids = str(len(vids))

    for i, vid in enumerate(vids):
        print('[' + str(i + 1) + '/' + nvids + ' @ ' + str(fps) + 'fps] extracting ' + vid)
        extract_av(vid, test_ann_dir, train_ann_dir, fps, eps, start_ts_filename, vid_save_path, extracted_path)


def normalise_mfccs(mfccs, apply_mean=True, apply_stddev=True):
    """
    Apply normalisation some cepstral mean variance normalisation (in part or in whole).
    """
    mean = 0
    stddev = 1

    if apply_mean:
        mean = mfccs.mean(axis=1).reshape(mfccs.shape[0], 1)
    if apply_stddev:
        stddev = mfccs.std(axis=1).reshape(mfccs.shape[0], 1)

    normalised = (mfccs - mean) / stddev
    return normalised, mean, stddev


def gen_mfcc(vid_id, extracted_path, train_ann_dir, test_ann_dir, stride, window_size, nmfcc, eps, apply_mean,
             apply_stddev):
    """
    Generate MFCCs for a video.
    """

    input_path = os.path.join(extracted_path, vid_id, 'audio.wav')
    pcm, sr = librosa.load(input_path, sr=None)
    sample_stride = int(sr * stride)
    window = int(sr * window_size)

    mfccs = librosa.feature.mfcc(pcm, sr, n_mfcc=nmfcc, n_fft=window, hop_length=sample_stride)
    normalised_mfccs, mean, stddev = normalise_mfccs(mfccs, apply_mean=apply_mean, apply_stddev=apply_stddev)

    ann_file = find_annotation_file(vid_id, train_ann_dir, test_ann_dir)
    mints, _ = get_min_max_timestamp(ann_file, eps=eps)

    new_ts = mints + window_size
    result = Mfcc(new_ts, nmfcc, mean, stddev,
                  window_size, stride, normalised_mfccs)

    return result


@main.command()
@click.argument('config-file', type=click.File('r'))
@click.argument('data-path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
def mfccs(config_file, data_path):
    """ Extract MFCCs from the dataset.

    CONFIG_FILE: the config file with settings for the experiment.
    DATA_PATH: the path to the folder with the data files.
    """

    config = read_config(config_file.name)
    extracted_path = os.path.join(data_path, config['extracted_path'])
    train_ann_dir = os.path.join(data_path, config['train_ann_dir'])
    test_ann_dir = os.path.join(data_path, config['test_ann_dir'])
    stride = config['stride']
    window_size = config['mfcc_window_size']
    nmfcc = config['nmfcc']
    eps = config['eps']
    apply_mean = config['apply_mean']
    apply_stddev = config['apply_stddev']

    dirs = os.listdir(extracted_path)
    ndirs = str(len(dirs))

    for i, vid_id in enumerate(dirs):
        print('[' + str(i + 1) + '/' + ndirs + '] generating MFCCs for ' + vid_id)
        mfccs = gen_mfcc(vid_id, extracted_path, train_ann_dir, test_ann_dir, stride, window_size, nmfcc, eps,
                         apply_mean, apply_stddev)
        output_path = os.path.join(extracted_path, vid_id, 'mfcc.pkl')
        pickle.dump(mfccs, open(output_path, 'wb'))


def get_vid_id(row):
    """
    Extract the video id from the annotations.
    """
    return row[0]


def get_bbox(row):
    """
    Extract the bounding box from the annotations.
    """
    return row[2], row[3], row[4], row[5]


def get_timestamp(row):
    """
    Extract the timestamp from the annotations.
    """
    return row[1]


def get_label(row):
    """
    Extract the active speaker label from the annotations.
    """
    return row[6]


def get_face_id(row):
    """
    Extract the face id from the annotations.
    """
    return row[7]


def row_to_annotation(row, width, height):
    """
    Convert row information to an annotation class.
    """
    vid_id = get_vid_id(row)
    timestamp = get_timestamp(row)
    bbox = get_bbox(row)
    label = get_label(row)
    face_id = get_face_id(row)
    face_size = get_face_size(bbox, width, height)
    return Annotation(vid_id, timestamp, bbox, label, face_id, face_size)


def generate_tracks(df, extracted_path):
    """
    Generate candidate face tracks from the annotations.
    """
    eps = 0.1
    firstrow = df.iloc[0]
    lastid = get_face_id(firstrow)
    lastts = get_timestamp(firstrow)

    vid_id = get_vid_id(firstrow)
    width, height = get_vid_width_height(extracted_path, vid_id)

    tracks = []
    track = []

    for i in range(len(df)):
        row = df.iloc[i]
        ann = row_to_annotation(row, width, height)

        if ann.face_id == lastid and ann.timestamp - lastts <= eps:
            track.append(ann)
        else:
            tracks.append(track)
            track = []
        lastid = ann.face_id
        lastts = ann.timestamp

    tracks.append(track)

    return tracks


def get_face_size(bbox, width, height):
    """
    Compute the face size in pixel coordinates.
    """

    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)
    x2 = int(bbox[2] * width)
    y2 = int(bbox[3] * height)
    width = x2 - x1
    height = y2 - y1
    return width, height


def closest_annotation(timestamp, track):
    """
    Find the closest annotation in a track to a given timestamp.
    """
    closest = 0
    closest_delta = float('inf')

    for i in range(len(track)):
        delta = abs(track[i].timestamp - timestamp)
        if delta < closest_delta:
            closest_delta = delta
            closest = i
    return track[closest]


def delete_small_tracks(tracks, window_size, fps):
    """
    Delete tracks that are smaller than the required window size. Use the timestamp to determine size rather than number of frames.
    """
    period = 1 / fps
    keep = []
    for track in tracks:
        filledtrack = []
        if len(track) == 0:
            continue
        ts_segment = track[-1].timestamp - track[0].timestamp
        if ts_segment <= window_size * period:
            continue

        hops = int(ts_segment / period)
        for i in range(hops):
            ca = closest_annotation(i * period + track[0].timestamp, track)
            filledtrack.append(ca)

        keep.append(filledtrack)

    return keep


def get_chunk(tracks, window_size, stride):
    """
    Cut tracks into window_size chunks. Uses a sliding window of stride 1.
    """
    chunks = []

    for track in tracks:
        n = len(track)
        for i in range(0, n - window_size, stride):
            if i + window_size < n:
                chunk = [track[i + j] for j in range(window_size)]
                chunks.append(chunk)

    return chunks


def get_homogenous(data):
    """
    Keep only homogenous data
    """
    keep = []
    for seq in data:
        label = seq[0].label
        hom = True
        for ann in seq:
            if ann.label != label:
                hom = False
                break
        if hom:
            keep.append(seq)

    return keep


def process_anns(ann_dir, output_filename, extracted_path, vid_frame_size, sequence_size, stride, fps):
    """
    Generate training meta data from the annotations in preparation for training.
    """
    window_size = int(vid_frame_size * sequence_size)

    data = []
    dirs = os.listdir(ann_dir)
    ndirs = len(dirs)
    for i, f in enumerate(dirs):
        print('[' + str(i + 1) + '/' + str(ndirs) + '] processing: ' + f)
        ann_file = os.path.join(ann_dir, f)
        df = pd.read_csv(ann_file, header=None)
        tracks = generate_tracks(df, extracted_path)
        tracks = delete_small_tracks(tracks, window_size, fps)
        chunk = get_chunk(tracks, window_size, stride=stride)
        data = data + chunk

    # Filter out inhomogenous sequences
    # data = get_homogenous(data)

    random.shuffle(data)

    serialised_data = pickle.dumps(data)
    with open(output_filename, 'wb') as f:
        f.write(serialised_data)


@main.command()
@click.argument('config-file', type=click.File('r'))
@click.argument('data-path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
def annotations(config_file, data_path):
    """ Generate training meta data from the annotations in preparation for training.

    CONFIG_FILE: the config file with settings for the experiment.
    DATA_PATH: the path to the folder with the data files.
    """

    config = read_config(config_file.name)
    extracted_path = os.path.join(data_path, config['extracted_path'])
    train_ann_dir = os.path.join(data_path, config['train_ann_dir'])
    test_ann_dir = os.path.join(data_path, config['test_ann_dir'])
    train_annotations_file = os.path.join(data_path, config['train_annotations_full'])
    test_annotations_file = os.path.join(data_path, config['test_annotations_full'])
    vid_frame_size = config['vid_frame_size']
    sequence_size = config['sequence_size']
    stride = config['ann_stride']
    fps = config['fps']

    print('Generating training annotations file.')
    process_anns(train_ann_dir, train_annotations_file, extracted_path, vid_frame_size, sequence_size, stride, fps)

    print('Generating test annotations file.')
    process_anns(test_ann_dir, test_annotations_file, extracted_path, vid_frame_size, sequence_size, stride, fps)


if __name__ == "__main__":
    main()
