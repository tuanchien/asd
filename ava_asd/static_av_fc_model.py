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

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, MaxPool2D, MaxPool3D, Input, Flatten, Dropout, concatenate
from tensorflow.keras.regularizers import l2


def audio_tower(shape=(13, 20, 1), drop_ratio=0.5, kreg=1e-4, weights=None):
    inputs = Input(shape=shape, name='a_in')  # input
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_a_conv1')(inputs)  # conv1
    x = MaxPool2D((1, 1), padding='same', name='s_av_a_pool1')(x)  # pool1
    x = Conv2D(192, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_a_conv2')(x)  # conv2
    x = MaxPool2D((3, 3), padding='same', name='s_av_a_pool2')(x)  # pool2
    x = Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_a_conv3')(x)  # conv3
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_a_conv4')(x)  # conv4
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_a_conv5')(x)  # conv5
    x = MaxPool2D((3, 3), padding='same', name='s_av_a_pool5')(x)  # pool5
    x = Conv2D(512, (5, 4), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_a_conv6')(x)  # conv6
    x = Flatten()(x)
    return inputs, x


def video_tower(shape=(5, 100, 100, 3), drop_ratio=0.5, kreg=1e-4, weights=None):
    inputs = Input(shape=shape, name='v_in')  # Input
    x = Conv3D(96, (5, 7, 7), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_v_conv1')(inputs)  # conv1
    x = MaxPool3D((1, 3, 3), padding='same', name='s_av_v_pool1')(x)  # pool1
    x = Conv3D(256, (1, 5, 5), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_v_conv2')(x)  # conv2
    x = MaxPool3D((1, 3, 3), padding='same', name='s_av_v_pool2')(x)  # pool2
    x = Conv3D(256, (1, 3, 3), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_v_conv3')(x)  # conv3
    x = Conv3D(256, (1, 3, 3), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_v_conv4')(x)  # conv4
    x = Conv3D(256, (1, 3, 3), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_v_conv5')(x)  # conv5
    x = MaxPool3D((1, 3, 3), padding='same', name='s_av_v_pool5')(x)  # pool5
    x = Conv3D(512, (1, 6, 6), padding='same', activation='relu', kernel_regularizer=l2(kreg),
               data_format='channels_last', name='s_av_v_conv6')(x)  # conv6
    x = Flatten()(x)
    return inputs, x


def static_av_fc(audio_shape=(13, 20, 1), video_shape=(5, 100, 100, 3), drop_ratio=0.5, num_classes=2, weights=None):
    # Define audio tower
    audio_input, audio_embedding = audio_tower(
        shape=audio_shape, drop_ratio=drop_ratio)
    aux_audio_out = Dense(num_classes, activation='softmax',
                          name='a_out')(audio_embedding)

    # Define video tower
    video_input, video_embedding = video_tower(
        shape=video_shape, drop_ratio=drop_ratio)
    aux_video_out = Dense(num_classes, activation='softmax',
                          name='v_out')(video_embedding)

    # AV prediction
    x = concatenate([audio_embedding, video_embedding])
    x = Dropout(drop_ratio)(x)

    main_output = Dense(num_classes, activation='softmax', name='main_out')(x)

    model = Model(inputs=[audio_input, video_input], outputs=[
        aux_audio_out, aux_video_out, main_output])

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model
