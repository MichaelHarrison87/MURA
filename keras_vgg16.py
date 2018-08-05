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

# Scripts created by me:
from models import VGG16
from utils import utils


start_time = time.time()

### INFO FOR TENSORBOARD
# Note: put the tensorboard log into a subdirectory of the main logs folder, i.e. /logs/run_1, /logs/run_2
# This lets tensorboard display their output as separate runs properly. For now we'll just automatically increment run number
dir_tensorboard_logs = "./tensorboard_logs/"
dir_tensorboard_logs = os.path.abspath(dir_tensorboard_logs)
num_tensorboard_runs = len(os.listdir(dir_tensorboard_logs))
dir_tensorboard_logs = dir_tensorboard_logs + "/run_" + str(num_tensorboard_runs+1)
# Note: make the log directory later, in case the code fails before the training step and the new directory is left empty

callback_tensorboard = callbacks.TensorBoard(log_dir=dir_tensorboard_logs, write_grads=True, write_images=True, histogram_freq=1)


### DATA PREP

# Images Directory
dir_images = "./data/processed/resized-100-77/" ## ENSURE CORRECT

# Get images paths & split training/validation
images_summary = pd.read_csv("./results/images_summary.csv")
images_summary_train = images_summary[images_summary.DataRole=="train"]
images_summary_valid = images_summary[images_summary.DataRole=="valid"]

filenames_relative_train = images_summary_train.FileName_Relative.values
filenames_train = dir_images + filenames_relative_train

filenames_relative_valid = images_summary_valid.FileName_Relative.values
filenames_valid = dir_images + filenames_relative_valid


# Get associated labels and convert to one-hot encoding
labels_train_scalar = images_summary_train.StudyOutcome.values
labels_train_onehot = to_categorical(labels_train_scalar)
num_classes = labels_train_onehot.shape[1]

labels_valid_scalar = images_summary_valid.StudyOutcome.values
labels_valid_onehot = to_categorical(labels_valid_scalar)

# Package train & valid items into dicts for easy reference/iteration
filenames_dict = {"train": filenames_train, "valid": filenames_valid}
labels_dict_scalar = {"train":labels_train_scalar, "valid":labels_valid_scalar}
labels_dict_onehot = {"train":labels_train_onehot, "valid":labels_valid_onehot}


# Check to_categorical working as intended (e.g. not differently for train vs valid, 0->[1,0], 1->[0,1])
# Need to ensure this as the categorical encoding (0->[1,0], 1->[0,1]) is assumed when handling predictions
# We check for mismatches element-wise across the entire list of labels, to ensure no reordering took place
for role in ['train','valid']:
    for x in range(2): # Iterate over outcomes 0/1
        mismatch = False

        # Compare orig & nohot labels
        labels_orig = images_summary[images_summary.DataRole==role].StudyOutcome.values
        labels_scalar = labels_dict_scalar[role]
        if sum(labels_orig!=labels_scalar)!=0:
            mismatch=True

        # Compare orig & one-hot labels
        labels_onehot = labels_dict_onehot[role]
        if x==0:
            labels_onehot_values = abs(labels_onehot[:,x]-1) # 0's are flagged 1 in the first one-hot column, so swap 0 to 1 and vice versa to match orig no-hot labels
        else:
            labels_onehot_values=labels_onehot[:,x] # The second one-hot column will match the original no-hot labels (i.e. 1's flagged as 1, 0's as 0)

        if sum(labels_orig!=labels_onehot_values)!=0:
            mismatch=True

        num_outcomes = sum(labels_dict_scalar[role]==x)
        num_outcomes_categorical = np.sum(labels_dict_onehot[role],axis=0)[x]

        if mismatch:
            print("WARNING: to_categorical not working as intended!")
            print("Num " + str(x) + " Raw (" + role + "):", num_outcomes)
            print("Num " + str(x) + " Categorical (" + role + "):", num_outcomes_categorical)
            exit()


# Get images dimension (all input images are same dimension, per pre-processing script)
test_image = Image.open(filenames_train[0])
image_width, image_height = test_image.size # PIL.Image gives size as (width, height)
image_depth = 3 # VGG16 requires 3-channel images

# TRAINING PARAMS
num_images_train = len(filenames_train)
num_images_valid = len(filenames_valid)
batch_size = 256
num_epochs = 1
num_steps_per_epoch = math.ceil(num_images_train/batch_size)  # Use entire dataset per epoch; round up to ensure entire dataset is covered if batch_size does not divide into num_images
num_steps_per_epoch_valid = math.ceil(num_images_valid/batch_size)   # As above

seed_train = 587
seed_valid = seed_train+1

# Now create the training & validation datasets
dataset_train = utils.create_dataset(filenames = filenames_train
, labels = labels_train_onehot
, num_channels = image_depth
, batch_size = batch_size
, shuffle_and_repeat = True
, repeat_count = num_epochs
, seed = seed_train)

dataset_valid = utils.create_dataset(filenames = filenames_valid
, labels = labels_valid_onehot
, num_channels = image_depth
, batch_size = batch_size
, shuffle_and_repeat = True
, repeat_count = num_epochs
, seed = seed_valid)


### TRAINING
# Open tensorflow session
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
config = tf.ConfigProto()
#config.log_device_placement=True

with tf.Session(config=config) as sess:
    sess.run(init)

    # Build the model
    model = VGG16.build_vgg16_notop(image_dimensions = (image_height, image_width, image_depth)
    , size_final_dense = 256
    , num_classes = num_classes
    , trainable=False)

    # Now train it
    model.compile(optimizer='RMSprop',loss='categorical_crossentropy', metrics=['accuracy'])

    train_start = time.time()
    #os.makedirs(os.path.dirname(dir_tensorboard_logs), exist_ok=True) # Make tensorboard log directory
    model.fit(dataset_train
    , epochs=num_epochs
    , steps_per_epoch=num_steps_per_epoch
    , validation_data=dataset_valid
    , validation_steps=num_steps_per_epoch_valid
    #, callbacks = [callback_tensorboard]
    )
    print("Training time: %s seconds" % (time.time() - train_start))
    print(model.summary())

    # Check that VGG16 Weights are unchanged, warn if not:
    vgg16_weights_sum = VGG16.sum_weights_vgg16_notop()
    vgg16_weights_post_train = utils.sum_model_weights(model)[1]
    if ((vgg16_weights_post_train/vgg16_weights_sum-1)>1E-6):
        print("WARNING: VGG Weights Updated During Training")

    print("TRAINING DONE")


    ### PREDICTIONS

    # Re-create the training & validation datasets without shuffling, so can match predictions with orig labels
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

print("--- %s seconds ---" % (time.time() - start_time))
