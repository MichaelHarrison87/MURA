import tensorflow as tf
import numpy as np
import pandas as pd
import os
import scipy.misc
import time


start_time = time.time()


images_summary = pd.read_csv("./results/images_summary.csv")
filenames = images_summary.FileName
filenames_relative = images_summary.FileName_Relative

# New Image Dimensions - ENSURE CORRECT
aspect_ratio = 1 # 1.3 is the most frequent height/width aspect ratio in the entire MURA dataset (~30% of all images; 1.2 & 1.4 were also each ~10% of images)
new_height = 200 # 139x139 is the smallest size accepted by inception_resnet_v2
new_width = round(new_height/aspect_ratio) # Want new width to be an integer
target_size = [new_height,new_width] # tf.resize_images takes size as [height,width]
dir_out = "./data/processed/resized-"+str(new_height)+"-"+str(new_width)+"-normalised-per-image" # ENSURE CORRECT

# Function below is appled to all items in the dataset via dataset.map
def process_image(filename):
  image_string = tf.read_file(filename)
  image = tf.image.decode_png(image_string)
  image = tf.image.per_image_standardization(image)
  image = tf.image.resize_images(image, size=target_size)
  return image

dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(process_image)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction=0.50
config.log_device_placement=True

with tf.Session(config=config) as sess:
    sess.run(init)

    i=0
    while True:
      try:
        image_example = sess.run(next_element)
        image_example = np.squeeze(image_example) # removes extraneous 3rd dimension (images are greyscale so have depth 1)

        path_out_rel = filenames_relative[i]
        path_out_abs = os.path.join(dir_out, path_out_rel)
        os.makedirs(os.path.dirname(path_out_abs), exist_ok=True)

        scipy.misc.imsave(path_out_abs, image_example)
        print("Image:",i+1,"of",filenames.shape[0])
        i+=1
      except tf.errors.OutOfRangeError:
        break


print("--- %s seconds ---" % (time.time() - start_time))
