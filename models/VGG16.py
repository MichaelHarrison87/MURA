"""
Functions & utilities for the VGG16 model
"""

import h5py
import numpy as np

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten

from tensorflow.python.keras.applications import VGG16

from utils import utils

def build_vgg16_notop(image_dimensions, pooling, size_final_dense, num_classes, trainable=False, weights='imagenet'):
    """
    Creates the VGG16 model without the final "top" dense layersself.
    Removing these top layers allows the VGG16 convolutional base to be used with a different set of image classes than the original VGG16 was trained on
    """

    vgg16_base = VGG16(weights=weights
    , include_top=False # Ignore the final dense layers, we'll train our own
    , input_shape=image_dimensions
    , pooling=pooling)
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

    # Location below should be relative to the script that calls this function (i.e. the directory above the models subdirectory), not this VGG16 script
    h5_file = "./models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    vgg16_h5 = h5py.File(h5_file)
    weights_sum = utils.sum_weights_h5_file(vgg16_h5)
    return(weights_sum)
