"""

This script contains the MDL model architecture.

Author: Makoto Ono (Based on Lu. et al (2022))

"""

import keras
from keras.layers import Conv2D, ConvLSTM2D, Dense, Flatten, BatchNormalization, Input
from keras.models import Model


def get_MDL_model(history_len, num_lanes, num_sections):
    speed_input = Input(shape=(history_len, num_lanes, num_sections, 1)) # history, lanes, sections, features
    speed_input1 = BatchNormalization()(speed_input)
    layer1 = ConvLSTM2D(
        filters=3,
        kernel_size=(3, 3),
        padding='same',
        kernel_regularizer = keras.regularizers.l2(0.01),
        bias_regularizer = keras.regularizers.l2(0.01),
        data_format='channels_last', 
        return_sequences=False)(speed_input1)
    layer1 = BatchNormalization()(layer1)
    layer2 = Conv2D(
        filters=1,
        kernel_size=(3, 3),
        data_format='channels_last',
        kernel_regularizer = keras.regularizers.l2(0.01),
        bias_regularizer = keras.regularizers.l2(0.01),
        activation='relu',
        padding='same')(layer1)
    flat1 = Flatten()(layer2)

    # UNCOMMENT THIS TO ADD DENSITY AND ACCELERATION AS INPUTS
    """
    dens_input = Input(shape=(history_len, num_lanes, num_sections, 1))
    dens_input1 = BatchNormalization()(dens_input)
    layer4 = ConvLSTM2D(
        filters=3,
        kernel_size=(3, 3),
        data_format='channels_last',
        kernel_regularizer = keras.regularizers.l2(0.01),
        bias_regularizer = keras.regularizers.l2(0.01),
        padding='same',
        return_sequences=False)(dens_input1)
    layer4 = BatchNormalization()(layer4)
    layer6 = Conv2D(
        filters=1,
        kernel_size=(3, 3),
        data_format='channels_last',
        kernel_regularizer = keras.regularizers.l2(0.01),
        bias_regularizer = keras.regularizers.l2(0.01),
        activation='relu',
        padding='same')(layer4)
    flat2 = Flatten()(layer6)

    
    acc_input = Input(shape=(history_len, 5, 21, 1))
    acc_input1 = BatchNormalization()(acc_input)
    layer7 = ConvLSTM2D(
        filters=3,
        kernel_size=(3, 3),
        data_format='channels_last',
        kernel_regularizer = keras.regularizers.l2(0.01),
        bias_regularizer = keras.regularizers.l2(0.01),
        padding='same',
        return_sequences=False)(acc_input1)
    layer7 = BatchNormalization()(layer7)
    layer8 = Conv2D(
        filters=1,
        kernel_size=(3, 3),
        data_format='channels_last',
        kernel_regularizer = keras.regularizers.l2(0.01),
        bias_regularizer = keras.regularizers.l2(0.01),
        activation='relu',
        padding='same')(layer7)
    flat3 = Flatten()(layer8)
    
    merged_output = keras.layers.concatenate([flat1, flat2])
    """
    out = Dense(num_lanes*num_sections)(flat1) # lanes, sections, pred_len
    return Model(inputs=[speed_input], outputs=out)
