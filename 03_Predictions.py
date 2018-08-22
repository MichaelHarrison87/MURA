import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image

from sklearn.metrics import confusion_matrix, cohen_kappa_score

from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import callbacks

# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.layers import Input, Dense, Flatten
# from tensorflow.python.keras.applications import inception_resnet_v2

# Scripts created by me:
from models import inception_resnet_v2
from utils import utils


start_time = time.time()


### PREDICTIONS

#Re-create the training & validation datasets without shuffling, so can match predictions with orig labels
    dataset_train_noshuffle = utils.create_dataset(filenames = filenames_train
    , labels = labels_train_onehot
    , num_channels = image_depth
    , batch_size = batch_size
    , shuffle_and_repeat = False)

    dataset_valid_noshuffle = utils.create_dataset(filenames = filenames_valid
    , labels = labels_valid_onehot
    , num_channels = image_depth
    , batch_size = batch_size
    , shuffle_and_repeat = False)

    # Get the predicted class probabilities, labels & probabilities of abnormality
    pred_probs_train, pred_labels_train, pred_probs_abnormal_train = utils.get_predictions(dataset=dataset_train_noshuffle
    , model=model
    , steps=num_steps_per_epoch)

    pred_probs_valid, pred_labels_valid, pred_probs_abnormal_valid = utils.get_predictions(dataset=dataset_valid_noshuffle
    , model=model
    , steps=num_steps_per_epoch_valid)

### Tensorflow no longer required, so come out of the session


# Calculate (image-wise) accuracy & cross-entropy loss
accuracy_train = utils.calc_accuracy(labels_train_scalar, pred_labels_train)
accuracy_valid = utils.calc_accuracy(labels_valid_scalar, pred_labels_valid)
loss_train = utils.calc_crossentropy_loss(labels_train_onehot, pred_probs_train)
loss_valid = utils.calc_crossentropy_loss(labels_valid_onehot, pred_probs_valid)
print("ACCURACY TRAIN:", accuracy_train)
print("ACCURACY VALID:", accuracy_valid)
print("LOSS TRAIN:", loss_train)
print("LOSS VALID:", loss_valid)


# Breakdowns of numbers of studies, from the orig MURA paper (Table 1, p3) - use these as a check on the numbers of studies we derive from our data
num_studies_published_dict = utils.get_num_studies_published()

studies_summary_train = utils.get_study_predictions(images_summary_train, pred_probs_abnormal_train)
studies_summary_valid = utils.get_study_predictions(images_summary_valid, pred_probs_abnormal_valid)

# Get the numbers of each study category derived from the data
num_studies_derived_dict = {"train":
    {"normal": np.sum(studies_summary_train.StudyOutcome==0)
    , "abnormal": np.sum(studies_summary_train.StudyOutcome==1)}
, "valid":
    {"normal": np.sum(studies_summary_valid.StudyOutcome==0)
    , "abnormal": np.sum(studies_summary_valid.StudyOutcome==1)}
}

# Now check these vs the published figures:
if(num_studies_published_dict != num_studies_derived_dict):
    print("NUMBER OF STUDIES MISMATCHED:")
    print("Published:", num_studies_published_dict)
    print("Derived:", num_studies_derived_dict)
    exit()

# Calc study-wise accuracy & other metrics
labels_study_train = studies_summary_train.StudyOutcome.values
pred_labels_study_train = studies_summary_train.PredLabel.values

labels_study_valid = studies_summary_valid.StudyOutcome.values
pred_labels_study_valid = studies_summary_valid.PredLabel.values

accuracy_study_train = utils.calc_accuracy(labels_study_train, pred_labels_study_train)
accuracy_study_valid = utils.calc_accuracy(labels_study_valid, pred_labels_study_valid)
print("accuracy_study_train:",accuracy_study_train)
print("accuracy_study_valid:",accuracy_study_valid)

# Confusion Matrices
confusion_matrix_train = confusion_matrix(labels_study_train, pred_labels_study_train)
confusion_matrix_valid = confusion_matrix(labels_study_valid, pred_labels_study_valid)

print("Confusion Matrix - Train:")
print(confusion_matrix_train)

print("Confusion Matrix - Valid:")
print(confusion_matrix_valid)

# Cohen's Kappa Score
cohen_kappa_train = cohen_kappa_score(labels_study_train, pred_labels_study_train)
cohen_kappa_valid = cohen_kappa_score(labels_study_valid, pred_labels_study_valid)

print("Cohen's Kappa Score - Train:")
print(cohen_kappa_train)

print("Cohen's Kappa Score - Valid:")
print(cohen_kappa_valid)
