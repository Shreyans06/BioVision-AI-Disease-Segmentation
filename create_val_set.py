import os
import random
import shutil

''' Splitting the train data into validation set '''

train_img_dir = "train_data/train"
train_mask_dir = "train_data/mask"

val_img_dir = "val_data/val"
val_mask_dir = "val_data/mask"

# Define the split ratio
split_ratio = 0.2
train_dir = os.listdir(train_img_dir)

# Randomly choose the validation list based on the split ratio
val_list = random.sample(train_dir , int(split_ratio * len(train_dir)))
val_images = set(val_list)

# Creating validation image directory
if not os.path.exists(val_img_dir):
    os.makedirs(val_img_dir)

# Creating the validation mask directory
if not os.path.exists(val_mask_dir):
    os.makedirs(val_mask_dir)

# Moving images from the train directory to the validation directory
for index , images in enumerate(val_images):
    shutil.move(os.path.join(train_img_dir , images) , val_img_dir)
    shutil.move(os.path.join(train_mask_dir , images) , val_mask_dir)
