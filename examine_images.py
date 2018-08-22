from PIL import Image
import numpy as np
import tensorflow as tf

"""
Inspect pixel values of raw vs processed images
"""
dir_raw = "./data/raw/"
dir_preprocessed = "./data/processed/resized-200-200-normalised-per-image/"
dir_trunk = "MURA-v1.1/train/XR_FOREARM/patient06749/study1_positive/"

path_image_raw = dir_raw + dir_trunk + "image1.png"
path_image_preprocessed = dir_preprocessed + dir_trunk + "image1.png"


### Raw
image_raw = Image.open( path_image_raw )
image_raw.load()
data_raw = np.asarray( image_raw, dtype="int8" ) # Default type of tf.decode_png is tf.unint8

print("Data Raw:")
print(data_raw)
print("shape:", data_raw.shape)
print("mean:", np.mean(data_raw))
print("stdev:", np.std(data_raw))
print("min:", np.min(data_raw))
print("max:", np.max(data_raw))

print()
print()

### Pre-Processed
image_preprocessed = Image.open( path_image_preprocessed )
image_preprocessed.load()
data_preprocessed = np.asarray( image_preprocessed, dtype="int8" ) # Default type of tf.decode_png is tf.unint8

print("Data Pre-Processed:")
print(data_preprocessed)
print("shape:", data_preprocessed.shape)
print("mean:", np.mean(data_preprocessed))
print("stdev:", np.std(data_preprocessed))
print("min:", np.min(data_preprocessed))
print("max:", np.max(data_preprocessed))
print()
print()

### Try Tensorflow
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
config = tf.ConfigProto()

with tf.Session(config=config) as sess:
    sess.run(init)

    image_string_raw = tf.read_file(path_image_raw)
    image_tf_raw = tf.image.decode_png(image_string_raw)
    image_tf_processed = tf.image.per_image_standardization(image_tf_raw)
    image_tf_processed = tf.image.resize_images(image_tf_processed, size=(200,200))

    image_tf_raw = sess.run(image_tf_raw)
    image_tf_processed = sess.run(image_tf_processed)

    print("Data Raw TF:")
    print(image_tf_raw)
    print("shape:", image_tf_raw.shape)
    print("mean:", np.mean(image_tf_raw))
    print("stdev:", np.std(image_tf_raw))
    print("min:", np.min(image_tf_raw))
    print("max:", np.max(image_tf_raw))
    print()
    print()

    print("Data Processed TF:")
    print(image_tf_processed)
    print("shape:", image_tf_processed.shape)
    print("mean:", np.mean(image_tf_processed))
    print("stdev:", np.std(image_tf_processed))
    print("min:", np.min(image_tf_processed))
    print("max:", np.max(image_tf_processed))
    print()
    print()

    image_string_preprocessed = tf.read_file(path_image_processed)
    image_tf_preprocessed = tf.image.decode_png(image_string_preprocessed)

    print("Data Pre-Processed TF:")
    print(image_tf_preprocessed)
    print("shape:", image_tf_processed.shape)
    print("mean:", np.mean(image_tf_preprocessed))
    print("stdev:", np.std(image_tf_preprocessed))
    print("min:", np.min(image_tf_preprocessed))
    print("max:", np.max(image_tf_preprocessed))
