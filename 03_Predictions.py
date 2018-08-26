import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image

from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_curve, roc_auc_score

from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.engine import InputLayer

# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.layers import Input, Dense, Flatten
# from tensorflow.python.keras.applications import inception_resnet_v2

# Scripts created by me:
from utils import utils

# NB: Image size must match those on which the model was trained on, as otherwise number of weights from flatten layer to dense layer will be incompatible

### GPU 1
os.environ["CUDA_VISIBLE_DEVICES"]="0"


start_time = time.time()

### Model
model_name = 'IRNV2_noweights_plateau_139_139_e_25' ## ENSURE CORRECT

### Prepare Dataset

# Images Directory
dir_images = "./data/processed/resized-139-139/" ## ENSURE CORRECT. Inception-ResNet-v2 min size is 139x139

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
print("Labels Valid Shape:", labels_valid_scalar.shape)

# Get images dimension (all input images are same dimension, per pre-processing script)
test_image = Image.open(filenames_valid[0])
image_width, image_height = test_image.size # PIL.Image gives size as (width, height)
image_depth = 3 # VGG16 requires 3-channel images
print("Image Dimensions:", image_height, image_width, image_depth)



# TRAINING PARAMS
num_images_valid = len(filenames_valid)
print("Num Images Valid:", num_images_valid)
batch_size = 32
num_epochs = 1
num_steps_per_epoch_valid = int(np.ceil(num_images_valid/batch_size))   # use np.ceil to ensure model predictions cover the whole validation set

seed_train = None #587
seed_valid = None #seed_train+1

# Now create the training & validation datasets
dataset_valid_noshuffle = utils.create_dataset(filenames = filenames_valid
, labels = labels_valid_onehot
, num_channels = 3
, batch_size = batch_size
, shuffle_and_repeat = False)
print("DATASETS CREATED")


### PREDICTIONS
# Open tensorflow session
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
config = tf.ConfigProto()
#config.log_device_placement=True

with tf.Session(config=config) as sess:
    sess.run(init)
    print("TF SESSION OPEN")

    # Load Model
    dir_model_saves = './keras_saves/'
    model = load_model(dir_model_saves + model_name + ".h5")

    print("MODEL LOADED")
    print(model.summary())
    print("Weights Sum:", utils.sum_model_weights(model)[1])

    # Get the predicted class probabilities, labels & probabilities of abnormality
    preds = model.predict(dataset_valid_noshuffle, steps=num_steps_per_epoch_valid)
    print("Preds Shape:", preds.shape)
    pred_probs_valid, pred_labels_valid, pred_probs_abnormal_valid = utils.get_predictions(dataset=dataset_valid_noshuffle
    , model=model
    , steps=num_steps_per_epoch_valid)
    print("Pred Labels Shape:", pred_labels_valid.shape)

### Tensorflow no longer required, so come out of the session


# Image-wise accuracy & cross-entropy loss
accuracy_valid = utils.calc_accuracy(labels_valid_scalar, pred_labels_valid)
loss_valid = utils.calc_crossentropy_loss(labels_valid_onehot, pred_probs_valid)
print("ACCURACY VALID:", accuracy_valid)
print("LOSS VALID:", loss_valid)

# Study-wise performence
# Breakdowns of numbers of studies, from the orig MURA paper (Table 1, p3) - use these as a check on the numbers of studies we derive from our data
num_studies_published_dict = utils.get_num_studies_published()

images_accuracy_check, studies_summary_valid = utils.get_study_predictions(images_summary_valid, pred_probs_abnormal_valid)

# Check image-wise accuracy to ensure image-wise results not reordered when calculating study-wise results
if(images_accuracy_check != accuracy_valid):
    print("Imagewise Accuracy Check Failed - Have Images Been Reordered?")
    print("ACCURACY VALID:", accuracy_valid)
    print("ACCURACY VALID CHECK:", images_accuracy_check)
    exit()

# Get the numbers of each study category derived from the data
num_studies_derived_dict = {"valid":
    {"normal": np.sum(studies_summary_valid.StudyOutcome==0)
    , "abnormal": np.sum(studies_summary_valid.StudyOutcome==1)}
}

# Now check these vs the published figures:
if(num_studies_published_dict["valid"] != num_studies_derived_dict["valid"]):
    print("NUMBER OF STUDIES MISMATCHED:")
    print("Published:", num_studies_published_dict)
    print("Derived:", num_studies_derived_dict)
    exit()

# Calc study-wise accuracy & other metrics
labels_study_valid = studies_summary_valid.StudyOutcome.values
pred_labels_study_valid = studies_summary_valid.PredLabel.values
pred_probs_study_valid = studies_summary_valid.PredProbAbnormal.values

accuracy_study_valid = utils.calc_accuracy(labels_study_valid, pred_labels_study_valid)
print("accuracy_study_valid:",accuracy_study_valid)

# Confusion Matrices
confusion_matrix_valid = confusion_matrix(labels_study_valid, pred_labels_study_valid)
print("Confusion Matrix - Valid:")
print(confusion_matrix_valid)

# Cohen's Kappa Score
cohen_kappa_valid = cohen_kappa_score(labels_study_valid, pred_labels_study_valid)
print("Cohen's Kappa Score - Valid:")
print(cohen_kappa_valid)

# ROC Curve & AUC
false_positive_valid, true_positive_valid, thresholds_valid = roc_curve(y_true=labels_study_valid, y_score=pred_probs_study_valid)
auc_valid = roc_auc_score(y_true=labels_study_valid, y_score=pred_probs_study_valid)
print("AUC - Valid:")
print(auc_valid)

# Save ROC Curve Data & AUC for visualisation (in Jupyter)
auc_valid_long = [auc_valid]*len(false_positive_valid)
roc_curve_df = pd.DataFrame({'FalsePositive':false_positive_valid, 'TruePositive':true_positive_valid, 'Thresholds':thresholds_valid, 'AUC': auc_valid_long})
dir_roc_curve = './results/roc_curve_'
roc_curve_df.to_csv(dir_roc_curve + model_name + '.csv', index=False)
print("ROC CURVE DATA SAVED")


print("--- %s seconds ---" % (time.time() - start_time))
