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

from tensorflow.keras.layers import Conv3D, Dense, MaxPool3D, Input, Flatten
from tensorflow.keras.models import Model


def static_video(shape=(5, 100, 100, 3), weights=None):
    """
    A static video model based on the description of arXiv:1906.10555v1 video tower, i.e., variation of VGGM.
    shape = (window size, height, width, chans)
    """
    inputs = Input(shape=shape, name='v_in')  # Input

    x = Conv3D(96, (5, 7, 7), padding='same', activation='relu',
               data_format='channels_last', name='s_v_conv1')(inputs)  # conv1
    x = MaxPool3D((1, 3, 3), padding='same', name='s_v_pool1')(x)  # pool1

    x = Conv3D(256, (1, 5, 5), padding='same', activation='relu',
               data_format='channels_last', name='s_v_conv2')(x)  # conv2
    x = MaxPool3D((1, 3, 3), padding='same', name='s_v_pool2')(x)  # pool2

    x = Conv3D(256, (1, 3, 3), padding='same', activation='relu',
               data_format='channels_last', name='s_v_conv3')(x)  # conv3

    x = Conv3D(256, (1, 3, 3), padding='same', activation='relu',
               data_format='channels_last', name='s_v_conv4')(x)  # conv4

    x = Conv3D(256, (1, 3, 3), padding='same', activation='relu',
               data_format='channels_last', name='s_v_conv5')(x)  # conv5
    x = MaxPool3D((1, 3, 3), padding='same', name='s_v_pool5')(x)  # pool5

    x = Conv3D(512, (1, 6, 6), padding='same', activation='relu',
               data_format='channels_last', name='s_v_conv6')(x)  # conv6

    x = Flatten()(x)
    x = Dense(256, activation='relu', name='s_v_fc7')(x)  # fc7
    outputs = Dense(2, activation='softmax', name='main_out')(x)  # fc7

    model = Model(inputs=inputs, outputs=outputs)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model
