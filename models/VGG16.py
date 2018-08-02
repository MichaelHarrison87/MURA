"""
Functions & utilities for the VGG16 model
"""

import h5py
import numpy as np

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten

from tensorflow.python.keras.applications import VGG16

def build_vgg16_notop(image_dimensions, size_final_dense, num_classes, trainable=False):
    """
    Creates the VGG16 model without the final "top" dense layers
    """

    vgg16_base = VGG16(weights='imagenet'
    , include_top=False # Ignore the final dense layers, we'll train our own
    , input_shape=image_dimensions)
    vgg16_base.trainable=trainable

    image_input = Input(shape=image_dimensions)

    x = vgg16_base(image_input)
    x = Flatten()(x)
    x = Dense(size_final_dense,activation='relu')(x)
    out = Dense(num_classes,activation='softmax')(x) # Task is classification

    model = Model(image_input, out)
    return(model)


def sum_weights_vgg16_notop():
    """
    Sums the weights of all layers in the vgg16_notop model.
    Used as a diagnostic to ensure these weights are not accidentally changed during training
    """
    vgg16_h5 = h5py.File("./models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

    keys = list(vgg16_h5.keys())

    weights_sum=0

    for k in keys:
        sub_keys = list(vgg16_h5.get(k).keys())

        for s in sub_keys:
            weights = vgg16_h5.get(k).get(s).value
            weights_sum += np.sum(weights)

    return(weights_sum)
