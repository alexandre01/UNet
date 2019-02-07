"""
Prepare the architecture of the workspace
"""

import os
import data_augmentation as data_aug
from distutils.dir_util import copy_tree
from glob import glob
from shutil import copy

IMAGES_DIR = "data/training/images/"
GT_DIR = "data/training/groundtruth/"
TEST_DIR = "data/test_set_images/"

#all generate path
MODELS = "save_model"
MODEL1 = "{}/model1/".format(MODELS)
MODEL2 = "{}/model2/".format(MODELS)

TEST_IMAGES = "data/test_images/"

TRAINING_AUGMENTED = "data/training_augmented"
AUGMENTED_IMAGES = "{}/images/".format(TRAINING_AUGMENTED)
AUGMENTED_GNDTRUTH = "{}/groundtruth/".format(TRAINING_AUGMENTED)

PREDICTIONS_TRAIN = "data/predictions_train"
PTRAIN_IMAGES = "{}/images/".format(PREDICTIONS_TRAIN)
PTRAIN_GNDTRUTH = "{}/groundtruth/".format(PREDICTIONS_TRAIN)

PREDICTIONS_TEST = "data/predictions_test/"

POST_PROCESSING = "data/post_processing/"

def main():
    #generate all the needed directories
    print("Create directories")
    if not os.path.exists(MODELS):
        os.makedirs(MODEL1)
        os.makedirs(MODEL2)
        
    if not os.path.exists(TEST_IMAGES):
        os.makedirs(TEST_IMAGES)
        
    if not os.path.exists(TRAINING_AUGMENTED):
        os.makedirs(AUGMENTED_IMAGES)
        os.makedirs(AUGMENTED_GNDTRUTH)
        
    if not os.path.exists(PREDICTIONS_TRAIN):
        os.makedirs(PTRAIN_IMAGES)
        os.makedirs(PTRAIN_GNDTRUTH)
        
    if not os.path.exists(PREDICTIONS_TEST):
        os.makedirs(PREDICTIONS_TEST)
        
    if not os.path.exists(POST_PROCESSING):
        os.makedirs(POST_PROCESSING)
        
    #generate augmented data
    print("Generate Augmented data")
    data_aug.data_augmentation(IMAGES_DIR,GT_DIR,AUGMENTED_IMAGES,AUGMENTED_GNDTRUTH)

    #copy augmented groundtruth in ptrain_gndtruth
    print("Copy groundtruth")
    copy_tree(AUGMENTED_GNDTRUTH,PTRAIN_GNDTRUTH)

    #flatten directory test_set_images
    for element in glob("{}test_*/*.png".format(TEST_DIR)):
        copy(element,TEST_IMAGES)
