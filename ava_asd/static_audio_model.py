# Author: Tuan Chien, James Diprose


from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Input, Flatten
from tensorflow.keras.models import Model


def static_audio(shape=(13, 20, 1), weights=None):
    """
    Implements the model architecture of the arXiv:1906.10555v1 audio feature tower, i.e., variation of VGGM.
    shape=(nmfcc, timewindow, 1)
    """
    inputs = Input(shape=shape, name='a_in')  # input

    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               data_format='channels_last', name='s_a_conv1')(inputs)  # conv1
    x = MaxPool2D((1, 1), padding='same', name='s_a_pool1')(x)  # pool1

    x = Conv2D(192, (3, 3), padding='same', activation='relu',
               data_format='channels_last', name='s_a_conv2')(x)  # conv2
    x = MaxPool2D((3, 3), padding='same', name='s_a_pool2')(x)  # pool2

    x = Conv2D(384, (3, 3), padding='same', activation='relu',
               data_format='channels_last', name='s_a_conv3')(x)  # conv3

    x = Conv2D(256, (3, 3), padding='same', activation='relu',
               data_format='channels_last', name='s_a_conv4')(x)  # conv4

    x = Conv2D(256, (3, 3), padding='same', activation='relu',
               data_format='channels_last', name='s_a_conv5')(x)  # conv5
    x = MaxPool2D((3, 3), padding='same', name='s_a_pool5')(x)  # pool5

    x = Conv2D(512, (5, 4), padding='same', activation='relu',
               data_format='channels_last', name='s_a_conv6')(x)  # conv6

    x = Flatten()(x)
    x = Dense(256, activation='relu', name='s_a_fc7')(x)  # fc7

    outputs = Dense(2, activation='softmax', name='main_out')(x)

    model = Model(inputs=inputs, outputs=outputs)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model
