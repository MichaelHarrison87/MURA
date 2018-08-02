# NB: Last run took approx 7mins on full MURA dataset

### Create Images Summary Table
# Create a table that summarises the MURA image data set
# Can then use this table to calculate summary statistics on the data
# e.g. # images or # studies per site, patient with most studies, # images per study etc
# Contains following fields:
# Data Role - Training/Validation Data
# Site - Elbow, Finder, Forearm etc
# Patient ID
# Study Number - per patient
# Image Number - per study
# Image Width - in pixels
# Image Height - in pixels
# Outcome - study positive/negative
#
# Most of the above fields are contained in the subfolder name in the MURA directory
# e.g. MURA-v1.1/train/XR_ELBOW/patient00011/study1_negative/image1.png
# So iterate over all images and parse the subfolder names to extract this info

import os
import pandas as pd
from PIL import Image
import time

start_time = time.time()

dir_Working = "./data/raw/"
dir_Working = os.path.abspath(dir_Working)
dir_MURA = os.path.join(dir_Working, "MURA-v1.1/")

summary_list = []

for root, dirs, files in os.walk(dir_MURA):
   for name in files:

       # Gives the subfolders within the MURA directory, for each file
       place_str = root.replace(dir_MURA, "")

       # Parse the subfolders to get the desired info above
       place_list = place_str.split("/")
       place_list += place_list[3].split("_")
       del place_list[3]
       place_list.append(name.replace(".png",""))

       # Get image dimensions
       filename = os.path.join(root, name)
       img = Image.open(filename)
       place_list += img.size

       path_abs = os.path.join(root, name)
       path_rel = os.path.relpath(path_abs, dir_Working)
       place_list.append(path_abs)
       place_list.append(path_rel)
       summary_list.append(place_list)

# Export Images Summary Table as csv
COLUMNS  = ["DataRole"
, "Site"
, "PatientID"
, "StudyNumber"
, "StudyOutcome"
, "ImageNumber"
, "ImageWidth"
, "ImageHeight"
, "FileName"
, "FileName_Relative"]
summary_table = pd.DataFrame(summary_list, columns = COLUMNS)
outcomes = outcomes = {"negative":0, "positive":1}
summary_table.StudyOutcome = summary_table.StudyOutcome.map(outcomes)

print("Table Shape:  ", summary_table.shape)

dir_output = "./results/"
summary_table.to_csv(dir_output+"images_summary.csv")

print("--- %s seconds ---" % (time.time() - start_time))
