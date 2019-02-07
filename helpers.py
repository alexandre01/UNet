import os
import numpy as np
import matplotlib.image as mpimg
import scipy.ndimage
import skimage.transform

"""
Helper function to read an image
If the image is 2d (no color channels), creates a new dimension to have same shape as colored images
"""
def imread(path):
    image = mpimg.imread(path)

    if len(image.shape) == 2:
        return image[:, :, np.newaxis]
    else:
        return image

def build_train_data(FLAGS,train_images_path,train_gt_path):
    print('Building training data...')
    train_data = np.empty((FLAGS.num_train, FLAGS.train_dim, FLAGS.train_dim, FLAGS.num_channels))
    train_labels = np.empty((FLAGS.num_train, FLAGS.train_dim, FLAGS.train_dim, 2))
    for i in range(FLAGS.num_train):
        train_image_path = os.path.join(train_images_path, 'satImage_{:03d}.png'.format(i + 1))
        groundtruth_path = os.path.join(train_gt_path, 'satImage_{:03d}.png'.format(i + 1))
        train_data[i,:,:,:] = imread(train_image_path)
        groundtruth = mpimg.imread(groundtruth_path)
        train_labels[i,:,:,0] = 1 - groundtruth
        train_labels[i,:,:,1] = groundtruth
    return train_data, train_labels

def build_test_data(FLAGS,test_path):
    print('Building testing data...')
    test_data = np.empty((FLAGS.num_test, FLAGS.test_dim, FLAGS.test_dim, FLAGS.num_channels))
    for i in range(FLAGS.num_test):
        test_img_path = os.path.join(test_path, 'test_{0}.png'.format(i + 1))
        test_data[i,:,:,:] = imread(test_img_path)
    return test_data

"""
Crops an image, given a position and a size
image: the image to crop
size: the size of the image to extract
y_offset and x_offset: the offsets, where to start cropping the image from the original one 
"""
def crop_image(image, size, y_offset, x_offset):

    is_2d = len(image.shape) == 2

    if is_2d:
        return image[y_offset:y_offset + size, x_offset:x_offset + size]
    else:
        return image[y_offset:y_offset + size, x_offset:x_offset + size, :]


"""
Rotates and crops and image, given a rotation-angle and the size of the image to return
angle: the angle of the rotation
size: the size of the image to return
"""
def rotate_crop_image(image, angle, size):
    h, w = image.shape[0], image.shape[1]

    rotated_image = scipy.ndimage.interpolation.rotate(image, angle, reshape=False, mode="nearest")

    y_offset = (h - size) // 2
    x_offset = (w - size) // 2

    return crop_image(rotated_image, size, y_offset, x_offset)


"""
Resizes an image to a given output size
image: the image to resize
size: the size of the rescaled image
"""
def resize_image(image, size):
    return skimage.transform.resize(image, (size, size), mode="reflect")
