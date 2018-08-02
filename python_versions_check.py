import tensorflow as tf
import os

# Note: on the GPU container (which uses Tesla K40c), could push the size of the test matrix to 30,000 (which ran in 18.5secs)
# Trying 35,000 gave an out-of-memory error

# Turn Off Tensorflow debuggin info:
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('\nTensorflow Version:')
print(tf.__version__)
print('~~~~~~\n')

# print('\nKeras Version:')
# import keras
# keras.__version__

print('\nTensorflow Device List:')
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('~~~~~~\n')


size = 30000


print('\nSpeed Test - GPU:')
import sys
import numpy as np
from datetime import datetime

device_name="/gpu:0"
shape=(int(size),int(size))

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

startTime = datetime.now()

config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction=0.50
config.log_device_placement=True
with tf.Session(config=config) as session:
        result = session.run(sum_operation)
        print(result)

print("\n" * 5)
print("Results - GPU:")
print("Shape:", shape, "Device:", device_name)
print("Matrix Size (GB):", 4*size*size/(10**9)) # nxn matrix of 4-byte floats, converted to GB
print("Time taken:", datetime.now() - startTime)

print('~~~~~~\n')



# print('\nSpeed Test - CPU:')
# device_name="/cpu:0"
# shape=(int(size),int(size))
#
# with tf.device(device_name):
#     random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
#     dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
#     sum_operation = tf.reduce_sum(dot_operation)
#
# startTime = datetime.now()
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
#         result = session.run(sum_operation)
#         print(result)
#
# print("\n" * 2)
# print("Shape:", shape, "Device:", device_name)
#
# print("Time taken:", datetime.now() - startTime)
#
# print("\n" * 2)
