import time
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow import keras
from PIL import Image

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import callbacks

import scipy.misc

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
dir_images = "./data/processed/resized-30-23/" ## ENSURE CORRECT

# Get images paths & split training/validation
images_summary = pd.read_csv("./results/images_summary.csv")
filenames_relative_train = images_summary[images_summary.DataRole=="train"].FileName_Relative.values
filenames_relative_valid = images_summary[images_summary.DataRole=="valid"].FileName_Relative.values
filenames_train = dir_images + filenames_relative_train
filenames_valid = dir_images + filenames_relative_valid

# Get associated labels
labels_train = images_summary[images_summary.DataRole=="train"].StudyOutcome.values
labels_train = to_categorical(labels_train)
num_classes = labels_train.shape[1]

labels_valid = images_summary[images_summary.DataRole=="valid"].StudyOutcome.values
labels_valid = to_categorical(labels_valid)

# Get images dimension (all input images are same dimension, per pre-processing script)
print(filenames_train[0])
test_image = Image.open(filenames_train[0])
image_height = test_image.size[0]
image_width = test_image.size[1]

# Function below applied to all images via tensorflow dataset.map() method
def read_image(filename, label):
  image_string = tf.read_file(filename)
  image = tf.image.decode_png(image_string, channels=1)
  return image, label

# Now create the training & validation datasets
dataset_train = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))
dataset_train = dataset_train.map(read_image)

dataset_valid = tf.data.Dataset.from_tensor_slices((filenames_valid, labels_valid))
dataset_valid = dataset_valid.map(read_image)

num_images_train = len(filenames_train)
num_images_valid = len(filenames_valid)

# TRAINING PARAMS
batch_size = 32
num_epochs = 5
num_steps_per_epoch = int(num_images_train/batch_size)  # Number of batches that constitutes an epoch
num_steps_per_epoch_valid = int(num_images_valid/batch_size)   # Number of batches that constitutes a validation epoch
# num_steps_per_epoch = 50
# num_steps_per_epoch_valid = 10

# Shuffle datasets & split into batches
seed_train = 587
seed_valid = seed_train+1

dataset_train = dataset_train.shuffle(num_images_train, seed=seed_train)
dataset_train = dataset_train.batch(batch_size).repeat()

dataset_valid = dataset_valid.shuffle(num_images_train, seed=seed_valid)
dataset_valid = dataset_valid.batch(batch_size).repeat()



### BUILD MODEL
def build_model():

    # Input Layer
    image_input = Input(shape=(image_height, image_width, 1)) # Final element is number of channels, set as 1 for greyscale

    # Convolutional Layer 1
    x = Conv2D( filters = 36
               , kernel_size = (3,3)
               , activation='relu'
               , padding='same' )(image_input)
    x = MaxPooling2D(pool_size = (2,2))(x)

    # Convolutional Layer 2
    x = Conv2D( filters = 36
               , kernel_size = (3,3)
               , activation='relu'
               , padding='same' )(x)
    x = MaxPooling2D(pool_size = (2,2))(x)

    # Convolutional Layer 3
    x = Conv2D( filters = 36
               , kernel_size = (3,3)
               , activation='relu'
               , padding='same' )(x)
    x = MaxPooling2D(pool_size = (2,2))(x)

    # Dense Layer
    x = Flatten()(x)
    x = Dense(30,activation='relu')(x)

    # Output Layer
    out =  Dense(num_classes,activation='softmax')(x) # Task is binary classification

    model = Model(image_input, out)
    return(model)


### TRAIN MODEL
# Open tensorflow session
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction=0.50
#config.log_device_placement=True

with tf.Session(config=config) as sess:
    sess.run(init)
    # from tensorflow.python.keras import backend as K
    # K.set_session(sess)

    # Now train the model
    model = build_model()
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


print("--- %s seconds ---" % (time.time() - start_time))
