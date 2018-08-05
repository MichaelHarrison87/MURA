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

# OLD:
def get_predictions(dataset, model, steps):
    """
    Gets the class probabilities, predicted classes & probability of abnormality calculated by a given model on a given tensorflow dataset
    The steps argument allows for iteration through the full set of data with a batched dataset (num steps = data_size/batch_size)
    """

    predicted_probs = model.predict(dataset, steps=steps)
    predicted_labels = np.argmax(predicted_probs, axis=1)
    predicted_probs_abnormal = predicted_probs[:,1] # probs that label=1, i.e. the x-ray is abnormal (the study was "positive")
    return predicted_probs, predicted_labels, predicted_probs_abnormal

# def get_predictions(class_probs):
#     """
#     Gets the predicted labels & prob of abnormality from a given set of class probabilities
#     """
#     labels = np.argmax(class_probs, axis=1)
#     probs_abnormal = class_probs[:,1] # probs that label=1, i.e. the x-ray is abnormal (the study was "positive")
#     return labels, probs_abnormal


def calc_accuracy(labels, predicted_labels):
    """
    Calculates the accuracy of a given set of predicted labels vs the actual class labels
    """
    num_obs = len(labels)
    accuracy = sum(predicted_labels==labels)/num_obs
    return accuracy


def calc_crossentropy_loss(class_vectors, predicted_class_probs):
    """
    Calculates the cross-entropy loss of a given set of predicted class probabilities vs actual class one-hot vectors
    Note cross-entropy is not symmetric - loss(P,Q) != loss(Q,P) so the order of arguments is important
    """
    # Add a small number to all values to avoid taking log(0)
    epsilon = 1e-7
    class_vectors = class_vectors + epsilon
    predicted_class_probs = predicted_class_probs + epsilon

    num_obs = len(class_vectors)
    entropy = -1 * class_vectors * np.log(predicted_class_probs) # Note np.log is natural log
    loss = np.sum(entropy)/num_obs
    return loss


def get_study_predictions(images_table, images_abnormal_probs, abnormal_threshold=0.5):
    """
    Given an images summary table and corresponding set of image-wise abnormality probabilities, produces study-wise abnormality probabilities and hence predicted labels for each study.
    images_table assumed to be a Pandas dataframe in the same format as the images_summary_table.csv
    images_abnormal_probs is a vector with same length as images_table
    abnormal_threshold allows a choice of threshold probability over which to predict "abnormal", albeit 0.5 is the most natural choice

    From the original MURA paper (pages 3/4):
    "We compute the overall probability of abnormality for the study by taking the arithmetic mean of the abnormality probabilities output by the network for each image.
    The model makes the binary prediction of abnormal if the probability of abnormality for the study is greater than 0.5."

    Studies are identified in the images_summary table by the combination of Site, PatientID & StudyNumber
    Some PatientID's appear in multiple Sites, while StudyNumber only counts studies per-patient - hence all 3 are required to uniquely identify a study
    We'll also include DataRole in the groupby in case we want to append train & validation sets back together
    (the other 3 columns are unqiue across train/valid so inclduing DataRole doesn't change # rows)
    """

    images_table["PredProbAbnormal"] = images_abnormal_probs
    studies_table = images_table.groupby(["DataRole", "Site", "PatientID", "StudyNumber", "StudyOutcome"], as_index=False).agg({"PredProbAbnormal":"mean"})
    num_studies = len(studies_table)

    # Calc study-wise predicted labels
    PredLabel = np.zeros(num_studies)
    PredLabel[studies_table["PredProbAbnormal"]>abnormal_threshold]=1
    studies_table["PredLabel"] = PredLabel

    return studies_table


# Breakdowns of numbers of studies, from the orig MURA paper (Table 1, p3) - use these as a check on the numbers of studies we derive from our data
def get_num_studies_published():
    num_studies_published_dict = {"train":
        {"normal": 8280
        , "abnormal": 5177}
    , "valid":
        {"normal": 661
        , "abnormal": 538}
    }
    num_studies_total_published = 14656

    num_studies_total_published_check = (num_studies_published_dict["train"]["normal"]
    + num_studies_published_dict["train"]["abnormal"]
    + num_studies_published_dict["valid"]["normal"]
    + num_studies_published_dict["valid"]["abnormal"])

    # Check totals of published studies, to protect vs typos in the numbers above
    if num_studies_total_published != num_studies_total_published_check:
        print("INPUT ERROR ON NUMBER OF STUDIES")
        print("num_studies_total_published", num_studies_total_published)
        print("num_studies_total_published_check", num_studies_total_published_check)
        exit()

    return num_studies_published_dict
