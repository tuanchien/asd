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

import yaml
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax, Nadam

from ava_asd.static_audio_model import static_audio
from ava_asd.static_av_fc_model import static_av_fc
from ava_asd.static_av_model import static_av
from ava_asd.static_video_model import static_video
from ava_asd.utils import get_window_sizes


def read_config(path):
    """
    Load the yaml file into a dictionary.
    """

    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_optimiser(config: dict):
    """
    Choose the correct optimiser.
    """

    # Read config
    optimiser = config['optimiser']
    learning_rate = config['learning_rate']

    if optimiser == 'SGD':
        return SGD(lr=learning_rate, nesterov=True, momentum=0.9, decay=1e-6)
    elif optimiser == 'Adam':
        return Adam(lr=learning_rate)
    elif optimiser == 'RMSprop':
        return RMSprop(lr=learning_rate)
    elif optimiser == 'Adagrad':
        return Adagrad(lr=learning_rate)
    elif optimiser == 'Adadelta':
        return Adadelta(lr=learning_rate)
    elif optimiser == 'Adamax':
        return Adamax(lr=learning_rate)
    elif optimiser == 'Nadam':
        return Nadam(lr=learning_rate)


def get_model(config: dict, **kwargs):
    """
    Select the correct model.
    """

    config_ = dict(config)

    # Kwargs is used to override values in the config
    for key, value in kwargs.items():
        config_[key] = value

    # Read config file
    mode = config_['mode']
    vid_frame_size = config_['vid_frame_size']
    mfcc_frame_size = config_['mfcc_frame_size']
    sequence_size = config_['sequence_size']
    width = config_['width']
    height = config_['height']
    nmfcc = config_['nmfcc']
    channels = config_['channels']
    classes = config_['classes']
    load_weights = config_['load_weights']
    weights_file = config_['weights_file']

    video_window_size, audio_window_size = get_window_sizes(mode, vid_frame_size, mfcc_frame_size, sequence_size)
    num_classes = len(set(classes.values()))

    loss = {}

    weights = None
    if load_weights:
        weights = weights_file

    if weights_file is None or not os.path.isfile(weights):
        weights = None

    if mode == 's_av':
        model = static_av(audio_shape=(nmfcc, audio_window_size, 1), video_shape=(
            video_window_size, height, width, channels), num_classes=num_classes, weights=weights)

    elif mode == 's_av_fc':
        model = static_av_fc(audio_shape=(nmfcc, audio_window_size, 1), video_shape=(
            video_window_size, height, width, channels), num_classes=num_classes, weights=weights)

    elif mode == 's_v':
        model = static_video(
            shape=(video_window_size, height, width, channels), weights=weights)

    elif mode == 's_a':
        model = static_audio(
            shape=(nmfcc, audio_window_size, 1), weights=weights)

    loss['main_out'] = 'categorical_crossentropy'
    loss['a_out'] = 'categorical_crossentropy'
    loss['v_out'] = 'categorical_crossentropy'

    return model, loss


def get_loss_weights(config: dict):
    """
    Create the correct loss weights dict.
    """

    # Read config file
    mode = config['mode']
    main_out_weight = config['main_out_weight']
    a_out_weight = config['a_out_weight']
    v_out_weight = config['v_out_weight']

    weights = {}
    weights['main_out'] = main_out_weight

    if mode == 's_av':
        weights['a_out'] = a_out_weight
        weights['v_out'] = v_out_weight

    return weights
