import os
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image

from helpers import *
import deconvnet as dnet
import mask_to_submission as msub
import init_archi as archi

flags = tf.app.flags

flags.DEFINE_integer('train_dim', 320, 'Dimension of training images.')
flags.DEFINE_integer('test_dim', 608, 'Dimension of testing images.')
flags.DEFINE_integer('num_train', 900, 'Number of training examples.')
flags.DEFINE_integer('num_test', 50, 'Number of testing examples.')
flags.DEFINE_integer('num_valid', 5, 'Number of validation examples.')
flags.DEFINE_integer('num_channels', 3, 'Number of channels of the images.')
flags.DEFINE_integer('num_epochs_M1', 100, 'Number of epochs when training the model1.')
flags.DEFINE_integer('num_epochs_M2', 10, 'Number of epochs when training the model2.')
flags.DEFINE_integer('batch_size', 5, 'Size of batches when doing optimization.')
flags.DEFINE_float('learning_rate', 5.0, 'Learning rate of the optimizer.')
flags.DEFINE_float('keep_prob', 0.5, 'The probability that a neuron\'s output is kept during dropout.')
flags.DEFINE_boolean('restore_M1', False, 'Whether or not to restore from a pre-trained model1.')
flags.DEFINE_boolean('restore_M2', False, 'Whether or not to restore from a pre-trained model2.')
flags.DEFINE_boolean('cont_train_M1', False, 'Whether to continue training of the model1 or not.')
flags.DEFINE_boolean('cont_train_M2', False, 'Whether to continue training of the model2 or not.')
flags.DEFINE_boolean('init_archi',False,'Whether or not to initiate the architecture of the workspace.')

FLAGS = flags.FLAGS

def main(_):
    #generate the architecture of the workspace if it is needed
    if FLAGS.init_archi:
        archi.main()
  
    #generate train_data and test_date for model1
    train_data, train_labels = build_train_data(FLAGS,archi.AUGMENTED_IMAGES,archi.AUGMENTED_GNDTRUTH)
    test_data = build_test_data(FLAGS,archi.TEST_IMAGES)
    
    #generate model1 and train
    option = dnet.Options(FLAGS.num_channels,FLAGS.num_epochs_M1,FLAGS.batch_size,FLAGS.learning_rate,FLAGS.keep_prob,archi.MODEL1)
    model = dnet.DeconvNet(train_data,train_labels,option)
    if FLAGS.restore_M1:
        model.restore()
    else:
        model.init()
        
    if not FLAGS.restore_M1 or FLAGS.cont_train_M1:
        model.train()

    # Predict on testing data and save images
    print("Predict on testing data with model1")
    predictions_test = model.predict(test_data)
    for i in range(FLAGS.num_test):
        Image.fromarray((predictions_test[i]*255).astype("uint8")).save(os.path.join(archi.PREDICTIONS_TEST, 'test_{}.png'.format(i+1)))
    predictions_test = build_test_data(FLAGS,archi.PREDICTIONS_TEST)
    
    # Predict on train data and save images if we need to train the second model
    if not FLAGS.restore_M2 or FLAGS.cont_train_M2:
        print("Predict on training data with model1")
        predictions_train = model.predict(train_data)
        for i in range(FLAGS.num_train):
            Image.fromarray((predictions_train[i]*255).astype("uint8")).save(os.path.join(archi.PTRAIN_IMAGES, 'satImage_{:03d}.png'.format(i + 1)))
        predictions_train, _ = build_train_data(FLAGS,archi.PTRAIN_IMAGES,archi.PTRAIN_GNDTRUTH) 
    
    #generate model2 and train
    option = dnet.Options(FLAGS.num_channels,FLAGS.num_epochs_M2,FLAGS.batch_size,FLAGS.learning_rate,FLAGS.keep_prob,archi.MODEL2)
    model = dnet.DeconvNet(predictions_train,train_labels,option)
    if FLAGS.restore_M2:
        model.restore()
    else:
        model.init()
        
    if not FLAGS.restore_M2 or FLAGS.cont_train_M2:
        model.train()
            
    # Predict on predictions_test data and save images
    print("Predict on testing data with model2")
    post_processing = model.predict(predictions_test)
    for i in range(FLAGS.num_test):
        Image.fromarray((post_processing[i]*255).astype("uint8")).save(os.path.join(archi.POST_PROCESSING, 'test_{}.png'.format(i+1)))
    
    # Create submission
    submission_filename = 'submission.csv'
    image_filenames = []
    for i in range(FLAGS.num_test):
        image_filename = archi.POST_PROCESSING + 'test_{}.png'.format(i+1)
        image_filenames.append(image_filename)
    msub.masks_to_submission(submission_filename, *image_filenames)

if __name__ == "__main__":
    tf.app.run()
   