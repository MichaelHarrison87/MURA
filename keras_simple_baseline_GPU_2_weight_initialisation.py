import time
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image

from sklearn.metrics import confusion_matrix, cohen_kappa_score

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.optimizers import RMSprop, Adam
#from tensorflow.python.keras.initializers import glorot_normal, he_normal

from tensorflow.python.keras.utils import to_categorical, multi_gpu_model
from tensorflow.python.keras import callbacks

# Scripts created by me:
from models import inception_resnet_v2
from utils import utils

### GPU 2
os.environ["CUDA_VISIBLE_DEVICES"]="1"

start_time = time.time()

### Model Name
model_name = "SimpleBaseline_Small_FromScratch_RMSProp_Default_Random_Normal_e_25_is_48_48" ## ENSURE CORRECT

# Images Directory
dir_images = "./data/processed/resized-48-48/" ## ENSURE CORRECT


### INFO FOR TENSORBOARD
# Note: put the tensorboard log into a subdirectory of the main logs folder, i.e. /logs/run_1, /logs/run_2
# This lets tensorboard display their output as separate runs properly. For now we'll just automatically increment run number
dir_tensorboard_logs = "./tensorboard_logs/"
dir_tensorboard_logs = os.path.abspath(dir_tensorboard_logs)
num_tensorboard_runs = len(os.listdir(dir_tensorboard_logs))
dir_tensorboard_logs = dir_tensorboard_logs + "/" + model_name
# Note: make the log directory later, in case the code fails before the training step and the new directory is left empty

callback_tensorboard = callbacks.TensorBoard(log_dir=dir_tensorboard_logs, write_grads=True, write_images=True, histogram_freq=1)


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
image_depth = 1 # VGG16 requires 3-channel images
print("Image Dimensions:", image_height, image_width, image_depth)

# TRAINING PARAMS
num_images_train = len(filenames_train)
num_images_valid = len(filenames_valid)
batch_size = 512
num_epochs = 25
num_steps_per_epoch = int(np.floor(num_images_train/batch_size))  # Use entire dataset per epoch; round up to ensure entire dataset is covered if batch_size does not divide into num_images
num_steps_per_epoch_valid = int(np.floor(num_images_valid/batch_size))   # As above

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
print("DATASETS CREATED")

### BUILD MODEL
# Big: initial_filters=256, size_final_dense=100
# Small: initial_filters=32, size_final_dense=100
def build_model(initial_filters, size_final_dense, initializer):

    # Input Layer
    image_input = Input(shape=(image_height, image_width, image_depth)) # Final element is number of channels, set as 1 for greyscale
    #x = BatchNormalization()(image_input)

    ### Block 1
    # Convolutional Layer 1
    x = Conv2D( filters = initial_filters
               , kernel_initializer = initializer
               , kernel_size = (3,3)
               , activation='relu'
               , padding='same' )(image_input)
    #x = BatchNormalization()(x)

    # Convolutional Layer 2
    x = Conv2D( filters = initial_filters
               , kernel_initializer = initializer
               , kernel_size = (3,3)
               , activation='relu'
               , padding='same' )(x)
    #x = BatchNormalization()(x)

    # Pooling Layer 1 - halve spatial dimension
    x = MaxPooling2D(pool_size = (2,2))(x)


    ### Block 2
    # Convolutional Layer 3 - double number of filters
    x = Conv2D( filters = initial_filters*2
               , kernel_initializer = initializer
               , kernel_size = (3,3)
               , activation='relu'
               , padding='same' )(x)
    #x = BatchNormalization()(x)

    # Convolutional Layer 4
    x = Conv2D( filters = initial_filters*2
               , kernel_initializer = initializer
               , kernel_size = (3,3)
               , activation='relu'
               , padding='same' )(x)
    #x = BatchNormalization()(x)

    # Pooling Layer 2 - halve spatial dimension
    x = MaxPooling2D(pool_size = (2,2))(x)


    ### Block 3
    # Convolutional Layer 5 - double number of filters
    x = Conv2D( filters = initial_filters*2*2
               , kernel_initializer = initializer
               , kernel_size = (3,3)
               , activation='relu'
               , padding='same' )(x)
    #x = BatchNormalization()(x)

    # Convolutional Layer 6
    x = Conv2D( filters = initial_filters*2*2
               , kernel_initializer = initializer
               , kernel_size = (3,3)
               , activation='relu'
               , padding='same' )(x)
    #x = BatchNormalization()(x)

    # Pooling Layer 3 - halve spatial dimension
    x = MaxPooling2D(pool_size = (2,2))(x)

    # Dense Layer
    x = Flatten()(x)
    x = Dense(size_final_dense
                , activation='relu'
                , kernel_initializer = initializer)(x)

    # Output Layer
    out =  Dense(num_classes
    , activation='softmax'
    , kernel_initializer = initializer)(x) # Task is binary classification

    model = Model(image_input, out)
    return(model)


### TRAINING
# Open tensorflow session
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
config = tf.ConfigProto()
#config.log_device_placement=True

with tf.Session(config=config) as sess:
    sess.run(init)
    print("TF SESSION OPEN")

    # Build the model
    model = build_model(initial_filters=32, size_final_dense=100, initializer = 'random_normal')
    print("MODEL BUILT")

    # Now train it
    opt_RMSprop = RMSprop()
    #callback_lr_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    model.compile(optimizer=opt_RMSprop,loss='categorical_crossentropy', metrics=['accuracy'])
    print("MODEL COMPILED")

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

    # Save the model
    dir_keras_saves = './keras_saves/'
    model.save(dir_keras_saves + model_name + ".h5")


    print("TRAINING DONE")

print("--- %s seconds ---" % (time.time() - start_time))
