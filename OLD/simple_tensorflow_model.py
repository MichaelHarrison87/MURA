import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image

from tensorflow.contrib.data import shuffle_and_repeat


start_time = time.time()

### INFO FOR TENSORBOARD
# Note: put the tensorboard log into a subdirectory of the main logs folder, i.e. /logs/run_1, /logs/run_2
# This lets tensorboard display their output as separate runs properly. For now we'll just automatically increment run number
dir_tensorboard_logs = "./tensorboard_logs/"
dir_tensorboard_logs = os.path.abspath(dir_tensorboard_logs)
num_tensorboard_runs = len(os.listdir(dir_tensorboard_logs))
dir_tensorboard_logs = dir_tensorboard_logs + "/run_" + str(num_tensorboard_runs+1)
# Note: make the log directory later, in case the code fails before the training step and the new directory is left empty

### DATA PREP

# Images Directory
dir_images = "./data/processed/resized-30-23/" ## ENSURE CORRECT

# Get images paths & split training/validation
images_summary = pd.read_csv("./results/images_summary.csv")
filenames_relative_train = images_summary[images_summary.DataRole=="train"].FileName_Relative.values
filenames_relative_valid = images_summary[images_summary.DataRole=="valid"].FileName_Relative.values
filenames_train = dir_images + filenames_relative_train
filenames_valid = dir_images + filenames_relative_valid
filenames_dict = {"train": filenames_train, "valid": filenames_valid}

# Get associated labels
num_classes = 2
labels_train = images_summary[images_summary.DataRole=="train"].StudyOutcome.values
#labels_train = tf.one_hot(labels_train, num_classes)

labels_valid = images_summary[images_summary.DataRole=="valid"].StudyOutcome.values
#labels_valid = tf.one_hot(labels_valid, num_classes)
labels_dict = {"train": labels_train, "valid": labels_valid}




# Get images dimension (all input images are same dimension, per pre-processing script)
test_image = Image.open(filenames_train[0])
image_height = test_image.size[0]
image_width = test_image.size[1]
image_depth = 1


# TRAINING PARAMS
batch_size = 32
num_epochs = 1
# num_steps_per_epoch = math.ceil(num_images_train/batch_size)  # Use entire dataset per epoch; round up to ensure entire dataset is covered if batch_size does not divide into num_images
# num_steps_per_epoch_valid = math.ceil(num_images_valid/batch_size)   # As above
num_steps_per_epoch = 50
num_steps_per_epoch_valid = 10

seed_train = 587
seed_valid = seed_train+1


# Function below applied to all images via tensorflow dataset.map() method
def read_image(filename, label):
  image_string = tf.read_file(filename)
  image = tf.image.decode_png(image_string, channels=image_depth)
  image = tf.cast(image, tf.float16)
  return image, label

# Function to create (train/valid) dataset and iterator
def create_dataset(role):
    filenames = filenames_dict[role]
    labels = labels_dict[role]
    num_images = len(filenames)

    # Read the images
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(read_image, num_parallel_calls=2)

    # Shuffle and batch the data
    dataset = dataset.apply(shuffle_and_repeat(buffer_size=num_images, count = num_epochs, seed= seed_train))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(num_images//batch_size)

    # iterator = dataset.make_one_shot_iterator()
    # images, labels = iterator.get_next()
    # images = {"x": images} # Put into a dict, since this is expected by the model functions

    return dataset


### BUILD MODEL
# Maybe separate the architecture from the training/evaluation parts
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features, [-1, image_height, image_width, image_depth])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=36,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=36,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.layers.flatten(pool2)
  dense = tf.layers.dense(inputs=pool2_flat, units=30, activation=tf.nn.relu)

  # Logits Layer
  logits = tf.layers.dense(inputs=dense, units=num_classes)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# ### TRAIN MODEL
# Open tensorflow session
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
config = tf.ConfigProto()
#config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    sess.run(init)

    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./model_checkpoints/simple_tf_model")

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: create_dataset("train"))

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: create_dataset("valid"),
        steps=None,
        start_delay_secs=10,  # Start evaluating after 10 sec.
        throttle_secs=5  # Evaluate only every 30 sec
    )

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


#     # Now train the model
#     #model = build_model()


print("--- %s seconds ---" % (time.time() - start_time))
