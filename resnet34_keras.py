import os
import warnings

from keras import layers
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import keras.regularizers as regularizers

def building_block(input_tensor, filters, stage, block, downsample=False):

    block_name_base = "conv" + str(stage) + "_x_"

    if downsample == True:
        # x = Conv2D(filters, (3, 3), strides=(2,2), padding="same",
        #     kernel_regularizer=regularizers.l2(0.001), name= block_name_base + block + "_conv1_ds")(input_tensor)
        # shortcut = Conv2D(filters, (3, 3), strides=(2,2), padding="same",
        #     kernel_regularizer=regularizers.l2(0.001), name=block_name_base + block + "_conv1_sc")(input_tensor)
        x = Conv2D(filters, (3, 3), strides=(2,2), padding="same", name= block_name_base + block + "_conv1_ds")(input_tensor)
        shortcut = Conv2D(filters, (3, 3), strides=(2,2), padding="same",
            name=block_name_base + block + "_conv1_sc")(input_tensor)

    else:
        # x = Conv2D(filters, (3, 3), strides=(1,1), padding="same",
        #     kernel_regularizer=regularizers.l2(0.01), name= block_name_base + block + "_conv1")(input_tensor)
        x = Conv2D(filters, (3, 3), strides=(1,1), padding="same", name= block_name_base + block + "_conv1")(input_tensor)
        shortcut = input_tensor
    x = BatchNormalization(axis=3, name=block_name_base + block + '_bn1')(x)
    x = Activation('relu')(x)

    # x = Conv2D(filters, (3, 3), strides=(1,1), padding="same",
    #     kernel_regularizer=regularizers.l2(0.001), name= block_name_base + block + "_conv2")(x)
    x = Conv2D(filters, (3, 3), strides=(1,1), padding="same", name= block_name_base + block + "_conv2")(x)
    x = BatchNormalization(axis=3, name=block_name_base + block + '_bn2')(x)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet34(inputs):

    # inputs = Input(shape=input_shape)
    # x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.001),name='conv1')(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = building_block(x, 64, 2, "a")
    x = building_block(x, 64, 2, "b")
    x = building_block(x, 64, 2, "c")

    x = building_block(x, 128, 3, "a", downsample=True)
    x = building_block(x, 128, 3, "b")
    x = building_block(x, 128, 3, "c")
    x = building_block(x, 128, 3, "d")


    x = building_block(x, 256, 4, "a", downsample=True)
    x = building_block(x, 256, 4, "b")
    x = building_block(x, 256, 4, "c")
    x = building_block(x, 256, 4, "d")
    x = building_block(x, 256, 4, "e")
    x = building_block(x, 256, 4, "f")

    x = building_block(x, 512, 5, "a", downsample=True)
    x = building_block(x, 512, 5, "b")
    x = building_block(x, 512, 5, "c")

    x = GlobalAveragePooling2D(name="global_avg_pool")(x)

    model = Model(inputs, x)
    return model
