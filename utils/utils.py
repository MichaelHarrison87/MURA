"""
Utility functions for use in other scripts
"""

import tensorflow as tf
import numpy as np
from PIL import Image


def read_image(filename, label, num_channels):
    """
    Decodes the given image from png into a numpy array
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string, channels=num_channels) # Represents images (height, width, channels)
    return image, label


def create_dataset(filenames, labels, num_channels, batch_size, shuffle_and_repeat, repeat_count=None, seed=None):
    """
    Function to create a Tensorflow Dataset from a given set of filenames & labels
    Setting shuffle_and_repeat to False will produce a dataset that allows only one iteration through, in the order of the original data
    This is useful for making predictions
    """

    num_images = len(filenames)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(lambda x, y: read_image(x, y, num_channels=num_channels))

    if shuffle_and_repeat:
        dataset = dataset.shuffle(buffer_size=num_images, reshuffle_each_iteration=True, seed=seed)
        dataset = dataset.repeat(repeat_count)

    dataset = dataset.batch(batch_size)
    return dataset


def sum_model_weights(model):
    """
    Diagnostic function that calcs the sum of all weights in a given model
    Used to check that (non-)trainable layers are (not) updating their weights during training, as expected
    """
    num_layers = len(model.layers)
    weights_by_layer = [None]*num_layers

    i=0
    for layer in model.layers:
        layer_weights = layer.get_weights() # This is a list of the layer's weight matrices
        weight_sum = 0

        # Iterate over the layer's weight matrices
        for w in layer_weights:
            weight_sum += np.sum(w)

        weights_by_layer[i] = weight_sum
        i+=1

    return(weights_by_layer)
