import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image

from sklearn.metrics import confusion_matrix, cohen_kappa_score

from tensorflow.python.keras.utils import to_categorical, multi_gpu_model
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.models import load_model

# Scripts created by me:
from models import inception_resnet_v2
from utils import utils


### GPU 2
os.environ["CUDA_VISIBLE_DEVICES"]="1"

start_time = time.time()

### Model Name
model_name = "IRNV2_noweights_lr_5E-5_plateau_L2_earlystopping_300_300_e_25" ## ENSURE CORRECT

# Images Directory
dir_images = "./data/processed/resized-300-300/" ## ENSURE CORRECT. Inception-ResNet-v2 min size is 139x139


### INFO FOR TENSORBOARD
# Note: put the tensorboard log into a subdirectory of the main logs folder, i.e. /logs/run_1, /logs/run_2
# This lets tensorboard display their output as separate runs properly. For now we'll just automatically increment run number
dir_tensorboard_logs = "./tensorboard_logs/"
dir_tensorboard_logs = os.path.abspath(dir_tensorboard_logs)
num_tensorboard_runs = len(os.listdir(dir_tensorboard_logs))
dir_tensorboard_logs = dir_tensorboard_logs + "/" + model_name
# Note: make the log directory later, in case the code fails before the training step and the new directory is left empty

callback_tensorboard = callbacks.TensorBoard(log_dir=dir_tensorboard_logs, write_grads=False, write_images=False, histogram_freq=1)


### DATA PREP

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
print("Num Classes:", num_classes)

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
print("Image Dimensions:", image_height, image_width, image_depth)

# TRAINING PARAMS
num_images_train = len(filenames_train)
num_images_valid = len(filenames_valid)
batch_size = 32
num_epochs = 25
num_steps_per_epoch = int(np.floor(num_images_train/batch_size))  # Use entire dataset per epoch; round up to ensure entire dataset is covered if batch_size does not divide into num_images
num_steps_per_epoch_valid = int(np.floor(num_images_valid/batch_size))   # As above

seed_train = None #587
seed_valid = None #seed_train+1

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
print("DATASETS CREATED")


### TRAINING
# Open tensorflow session
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
config = tf.ConfigProto()
#config.log_device_placement=True

with tf.Session(config=config) as sess:
    sess.run(init)
    print("TF SESSION OPEN")

    # Build the model
    model = inception_resnet_v2.build_inception_resnet_v2_notop(image_dimensions = (image_height, image_width, image_depth)
    , pooling = None
    , size_final_dense = 256
    , num_classes = num_classes
    , trainable = True
    , weights = None
    , final_regulariser = l2(0.002))
    #model = multi_gpu_model(model, gpus=2)
    print("MODEL BUILT")

    # Now train it
    opt_RMSprop = RMSprop(lr=0.00005)
    model.compile(optimizer=opt_RMSprop,loss='categorical_crossentropy', metrics=['accuracy'])
    callback_lr_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    callback_earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=4)
    print("MODEL COMPILED")

    train_start = time.time()
    os.makedirs(os.path.dirname(dir_tensorboard_logs), exist_ok=True) # Make tensorboard log directory
    model.fit(dataset_train
    , epochs=num_epochs
    , steps_per_epoch=num_steps_per_epoch
    , validation_data=dataset_valid
    , validation_steps=num_steps_per_epoch_valid
    , callbacks = [callback_tensorboard, callback_lr_plateau, callback_earlystopping]
    )
    print("Training time: %s seconds" % (time.time() - train_start))
    print(model.summary())


    # Save the model
    dir_keras_saves = './keras_saves/'
    model.save(dir_keras_saves + model_name + ".h5")


    #Check that inception_resnet_v2 Weights are unchanged, warn if not:
    inception_resnet_v2_weights_sum = inception_resnet_v2.sum_weights_inception_resnet_v2_notop()
    inception_resnet_v2_weights_post_train = utils.sum_model_weights(model)[1]
    if (abs(inception_resnet_v2_weights_post_train/inception_resnet_v2_weights_sum-1)>1E-6):
        print("WARNING: Inception-ResNet-v2 Weights Updated During Training")

    print("TRAINING DONE")


print("--- %s seconds ---" % (time.time() - start_time))
