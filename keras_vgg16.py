import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image

from sklearn.metrics import confusion_matrix

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
filenames_relative_train = images_summary[images_summary.DataRole=="train"].FileName_Relative.values
filenames_relative_valid = images_summary[images_summary.DataRole=="valid"].FileName_Relative.values
filenames_train = dir_images + filenames_relative_train
filenames_valid = dir_images + filenames_relative_valid
filenames_dict = {"train": filenames_train, "valid": filenames_valid}

# Get associated labels and convert to one-hot encoding
labels_train_nohot = images_summary[images_summary.DataRole=="train"].StudyOutcome.values
labels_train = to_categorical(labels_train_nohot)
num_classes = labels_train.shape[1]

labels_valid_nohot = images_summary[images_summary.DataRole=="valid"].StudyOutcome.values
labels_valid = to_categorical(labels_valid_nohot)

# Package train & valid labels into a dict for easy reference/iteration
labels_dict = {"train":labels_train, "valid":labels_valid}
labels_dict_nohot = {"train":labels_train_nohot, "valid":labels_valid_nohot} # Original "no hot" 0/1 label encoding


# Check to_categorical working as intended (e.g. not differently for train vs valid, 0->[1,0], 1->[0,1])
# Need to ensure this as the categorical encoding (0->[1,0], 1->[0,1]) is assumed when handling predictions
# We check for mismatches element-wise across the entire list of labels, to ensure no reordering took place
for role in ['train','valid']:
    for x in range(2): # Iterate over outcomes 0/1
        mismatch = False

        # Compare orig & nohot labels
        labels_orig = images_summary[images_summary.DataRole==role].StudyOutcome.values
        labels_nohot = labels_dict_nohot[role]
        if sum(labels_orig!=labels_nohot)!=0:
            mismatch=True

        # Compare orig & one-hot labels
        labels_onehot = labels_dict[role]
        if x==0:
            labels_onehot_values = abs(labels_onehot[:,x]-1) # 0's are flagged 1 in the first one-hot column, so swap 0 to 1 and vice versa to match orig no-hot labels
        else:
            labels_onehot_values=labels_onehot[:,x] # The second one-hot column will match the original no-hot labels (i.e. 1's flagged as 1, 0's as 0)

        if sum(labels_orig!=labels_onehot_values)!=0:
            mismatch=True

        num_outcomes = sum(labels_dict_nohot[role]==x)
        num_outcomes_categorical = np.sum(labels_dict[role],axis=0)[x]

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
batch_size = 32
num_epochs = 1
num_steps_per_epoch = math.ceil(num_images_train/batch_size)  # Use entire dataset per epoch; round up to ensure entire dataset is covered if batch_size does not divide into num_images
num_steps_per_epoch_valid = math.ceil(num_images_valid/batch_size)   # As above
# num_steps_per_epoch = 10
# num_steps_per_epoch_valid = 1

seed_train = 587
seed_valid = seed_train+1

# Now create the training & validation datasets
dataset_train = utils.create_dataset(filenames = filenames_train
, labels = labels_train
, num_channels = image_depth
, batch_size = batch_size
, shuffle_and_repeat = True
, repeat_count = num_epochs
, seed = seed_train)

dataset_valid = utils.create_dataset(filenames = filenames_valid
, labels = labels_valid
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
    model.compile(optimizer='RMSprop',loss='binary_crossentropy', metrics=['accuracy'])

    train_start = time.time()
    os.makedirs(os.path.dirname(dir_tensorboard_logs), exist_ok=True) # Make tensorboard log directory
    model.fit(dataset_train
    , epochs=num_epochs
    , steps_per_epoch=num_steps_per_epoch
    , validation_data=dataset_valid
    , validation_steps=num_steps_per_epoch_valid
    , callbacks = [callback_tensorboard]
    )
    print("Training time: %s seconds" % (time.time() - train_start))
    print(model.summary())

    # Check that VGG Weights are unchanged, warn if not:
    vgg16_weights_sum = VGG16.sum_weights_vgg16_notop()
    vgg16_weights_post_train = utils.sum_model_weights(model)[1]
    if ((vgg16_weights_post_train/vgg16_weights_sum-1)>1E-6):
        print("WARNING: VGG Weights Updated During Training")

    print("TRAINING DONE")
    exit()

### PREDICTIONS
# Model will output predictions for each image; need to aggregate into predictions for each study (composed of multiple images)
# Note the predctions are binary one-hot vectors (0->[1,0], 1->[0,1]) in line with the original labels
# So multiplying labels & predictions will give 1 for correct prediction and 0 for incorrect
# model.predict shuffles the predictions?

    # Now re-create the training & validation datasets without shuffling, so can match predictions with orig labels
    dataset_train_noshuffle = create_dataset(role="train", batch_size=batch_size, shuffle_and_repeat=False)
    dataset_valid_noshuffle = create_dataset(role="valid", batch_size=batch_size, shuffle_and_repeat=False)

    predictions_train  = model.predict(dataset_train_noshuffle, steps = num_steps_per_epoch)
    predictions_valid  = model.predict(dataset_valid_noshuffle, steps = num_steps_per_epoch_valid)
    predictions_dict = {"train": predictions_train, "valid": predictions_valid}

    # Convert predictions to "no-hot" 0/1 representations
    predictions_train_nohot = np.argmax(predictions_train, axis=1)
    predictions_valid_nohot = np.argmax(predictions_valid, axis=1)

    # Package into dict
    predictions_dict = {"train": predictions_train, "valid": predictions_valid}
    predictions_dict_nohot = {"train": predictions_train_nohot, "valid": predictions_valid_nohot}

    accuracy_train = sum(predictions_train_nohot==labels_train_nohot)/num_images_train
    accuracy_train_shuffle = sum(predictions_train_shuffle_nohot==labels_train_nohot)/num_images_train
    accuracy_valid = sum(predictions_valid_nohot==labels_valid_nohot)/num_images_valid
    accuracy_valid_shuffle = sum(predictions_valid_shuffle_nohot==labels_valid_nohot)/num_images_valid
    accuracy_test = sum(predictions_test_full_nohot==labels_test_full_nohot)/num_images_train
    print("ACCURACY TRAIN:", accuracy_train)
    print("ACCURACY TRAIN SHUFFLE:", accuracy_train_shuffle)
    print("ACCURACY VALID:", accuracy_valid)
    print("ACCURACY VALID SHUFFLE:", accuracy_valid_shuffle)
    print("ACCURACY TEST:",accuracy_test)


    # predictions_valid  = model.predict(dataset_valid_noshuffle, steps = num_steps_per_epoch_valid)
    # print("SHAPE predictions_valid:", predictions_valid.shape)
    # predictions_dict = {"train": predictions_train, "valid": predictions_valid}
    #
    # # Calc accuracy
    # for role in ["train", "valid"]:
    #     predictions = predictions_dict[role]
    #     labels = labels_dict[role]
    #     accuracy_vector = predictions * labels
    #     accuracy = np.sum(accuracy_vector)/predictions.shape[0]
    #     print("Accuracy - "+role, accuracy)
    #
    # # Convert one-hot predictions to "no-hot" 0/1 encoding
    # predictions_train = np.argmax(predictions_train, axis=1)
    # predictions_valid = np.argmax(predictions_valid, axis=1)
    # predictions_dict = {"train": predictions_train, "valid": predictions_valid}
    #
    # # Calculate confusion matrices
    # for role in ["train", "valid"]:
    #     labels = labels_dict_nohot[role]
    #     predictions = predictions_dict[role]
    #
    #     cm = confusion_matrix(labels, predictions)
    #     print("Confusion Matrix - " + role)
    #     print(cm)
    #     print("Accuracy:", (cm[0,0]+cm[1,1])/np.sum(cm))






print("--- %s seconds ---" % (time.time() - start_time))
