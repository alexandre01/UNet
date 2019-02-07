"""
Data augmentation: in order to avoid overfitting, we can artificially increase to number of input images by
generating new ones from the 100 training images
"""

from scipy import misc
import os
import random
from helpers import crop_image, rotate_crop_image, resize_image

# Parameters
ANGLES = [-90., -45., 45., 90.]
IMAGE_SIZE = 400
NEW_IMAGE_SIZE = 320

def data_augmentation(IMAGES_DIR,GT_DIR,AUG_IMAGES_DIR,AUG_GT_DIR):
    image_names = os.listdir(IMAGES_DIR)

    for i, image_name in enumerate(image_names):
        image = misc.imread(IMAGES_DIR + image_name)
        gt_image = misc.imread(GT_DIR + image_name)

        # First, saving the original image into the new folder (after rescaling to the new image size)
        misc.imsave(AUG_IMAGES_DIR + image_name, resize_image(image, NEW_IMAGE_SIZE))
        misc.imsave(AUG_GT_DIR + image_name, resize_image(gt_image, NEW_IMAGE_SIZE))

        # Generating 4 cropped images from the original one
        for y in range(2):
            y_offset = y * (IMAGE_SIZE - NEW_IMAGE_SIZE)

            for x in range(2):
                x_offset = x * (IMAGE_SIZE - NEW_IMAGE_SIZE)

                cropped_image = crop_image(image, NEW_IMAGE_SIZE, y_offset, x_offset)
                gt_cropped_image = crop_image(gt_image, NEW_IMAGE_SIZE, y_offset, x_offset)

                # Keep same name convention, e.g. for first image, the names will be
                # satImage_101, satImage_201, satImage_301 and satImage_401
                new_image_name = "satImage_{:03d}.png".format(y * 200 + (x + 1) * 100 + (i + 1))

                misc.imsave(AUG_IMAGES_DIR + new_image_name, cropped_image)
                misc.imsave(AUG_GT_DIR + new_image_name, gt_cropped_image)

        # Generating 4 rotated images from the original one
        for j, angle in enumerate(ANGLES):

            # Randomly add a small variation to the defined angle: angle' = angle +/- 5 degrees
            angle += 5 * (random.random() - 0.5)

            rotated_image = rotate_crop_image(image, angle, NEW_IMAGE_SIZE)
            gt_rotated_image = rotate_crop_image(gt_image, angle, NEW_IMAGE_SIZE)

            new_image_name = "satImage_{:03d}.png".format(500 + j * 100 + (i + 1))

            misc.imsave(AUG_IMAGES_DIR + new_image_name, rotated_image)
            misc.imsave(AUG_GT_DIR + new_image_name, gt_rotated_image)