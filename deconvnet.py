import os
import numpy as np
import tensorflow as tf

class Options(object):
    """Parameters used by the DeconvNet model."""
    def __init__(self, num_channels, num_epochs, batch_size, learning_rate, keep_prob, save_path):
        super(Options, self).__init__()
        self.num_channels = num_channels
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.save_path = save_path
        
    

class DeconvNet(object):
    """The DeconvNet model."""
    def __init__(self, train_data, train_labels, options):
        super(DeconvNet, self).__init__()
        self.train_data = train_data
        self.train_labels = train_labels
        self.options = options

        self.build_graph()
        self.session = tf.Session(graph = self.graph)

    def __del__(self):
        self.session.close()
        print('TensorFlow session for DeconvNet is closed.')

    def build_graph(self):
        print('Building TensorFlow graph for DeconvNet...')
        opts = self.options

        def weight_variable(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

        def bias_variable(shape):
            return tf.Variable(tf.constant(0.1, shape = shape))

        def conv(x, W_shape, b_shape, stride = 1, padding = 'SAME', bn = True, activation = True, dropout = False):
            W = weight_variable(W_shape)
            b = bias_variable(b_shape)
            layer = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding) + b
            if bn:
                layer = tf.layers.batch_normalization(layer, training = self.phase)
            if activation:
                layer = tf.nn.relu(layer)
            if dropout:
                layer = tf.nn.dropout(layer, self.keep_prob)
            return layer

        def deconv(x, W_shape, b_shape, output_shape, stride = 1, padding = 'SAME', dropout = False):
            W = weight_variable(W_shape)
            b = bias_variable(b_shape)
            layer = tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, stride, stride, 1], padding = padding) + b
            layer = tf.layers.batch_normalization(layer, training = self.phase)
            layer = tf.nn.relu(layer)
            if dropout:
                layer = tf.nn.dropout(layer, self.keep_prob)
            return layer

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, shape = [None, None, None, opts.num_channels])
            self.y = tf.placeholder(tf.float32, shape = [None, None, None, 2])
            self.phase = tf.placeholder(tf.bool)
            self.keep_prob = tf.placeholder(tf.float32)

            conv_1_1 = conv(self.x, [3, 3, opts.num_channels, 64], [64])
            conv_1_2 = conv(conv_1_1, [3, 3, 64, 64], [64], stride = 2)

            conv_2_1 = conv(conv_1_2, [3, 3, 64, 128], [128])
            conv_2_2 = conv(conv_2_1, [3, 3, 128, 128], [128], stride = 2)

            conv_3_1 = conv(conv_2_2, [3, 3, 128, 256], [256])
            conv_3_2 = conv(conv_3_1, [3, 3, 256, 256], [256])
            conv_3_3 = conv(conv_3_2, [3, 3, 256, 256], [256], stride = 2)

            conv_4_1 = conv(conv_3_3, [3, 3, 256, 512], [512])
            conv_4_2 = conv(conv_4_1, [3, 3, 512, 512], [512])
            conv_4_3 = conv(conv_4_2, [3, 3, 512, 512], [512], stride = 2)

            conv_5_1 = conv(conv_4_3, [3, 3, 512, 512], [512])
            conv_5_2 = conv(conv_5_1, [3, 3, 512, 512], [512])
            conv_5_3 = conv(conv_5_2, [3, 3, 512, 512], [512], stride = 2)

            conv_6_1 = conv(conv_5_3, [3, 3, 512, 512], [512])
            conv_6_2 = conv(conv_6_1, [3, 3, 512, 512], [512])
            conv_6_3 = conv(conv_6_2, [3, 3, 512, 512], [512], stride = 2)

            deconv_6_3 = deconv(conv_6_3, [3, 3, 512, 512], [512], tf.shape(conv_6_2), stride = 2)
            deconv_6_2 = deconv(deconv_6_3, [3, 3, 512, 512], [512], tf.shape(conv_6_1))
            deconv_6_1 = deconv(deconv_6_2, [3, 3, 512, 512], [512], tf.shape(conv_5_3))

            deconv_5_3 = deconv(deconv_6_1, [3, 3, 512, 512], [512], tf.shape(conv_5_2), stride = 2)
            fuse_5 = tf.add(deconv_5_3, conv_5_2)
            deconv_5_2 = deconv(fuse_5, [3, 3, 512, 512], [512], tf.shape(conv_5_1))
            deconv_5_1 = deconv(deconv_5_2, [3, 3, 512, 512], [512], tf.shape(conv_4_3))

            deconv_4_3 = deconv(deconv_5_1, [3, 3, 512, 512], [512], tf.shape(conv_4_2), stride = 2)
            fuse_4 = tf.add(deconv_4_3, conv_4_2)
            deconv_4_2 = deconv(fuse_4, [3, 3, 512, 512], [512], tf.shape(conv_4_1))
            deconv_4_1 = deconv(deconv_4_2, [3, 3, 256, 512], [256], tf.shape(conv_3_3), dropout = True)

            deconv_3_3 = deconv(deconv_4_1, [3, 3, 256, 256], [256], tf.shape(conv_3_2), stride = 2)
            fuse_3 = tf.add(deconv_3_3, conv_3_2)
            deconv_3_2 = deconv(fuse_3, [3, 3, 256, 256], [256], tf.shape(conv_3_1))
            deconv_3_1 = deconv(deconv_3_2, [3, 3, 128, 256], [128], tf.shape(conv_2_2), dropout = True)

            deconv_2_2 = deconv(deconv_3_1, [3, 3, 128, 128], [128], tf.shape(conv_2_1), stride = 2)
            fuse_2 = tf.add(deconv_2_2, conv_2_1)
            deconv_2_1 = deconv(fuse_2, [3, 3, 64, 128], [64], tf.shape(conv_1_2), dropout = True)

            deconv_1_2 = deconv(deconv_2_1, [3, 3, 64, 64], [64], tf.shape(conv_1_1), stride = 2)
            shape = tf.shape(self.x)
            deconv_shape = [shape[0], shape[1], shape[2], 64]
            deconv_1_1 = deconv(deconv_1_2, [3, 3, 64, 64], [64], deconv_shape)

            self.convs = [conv_1_2, conv_2_2, conv_3_3, conv_4_3, conv_5_3, conv_6_3]
            self.deconvs = [deconv_6_1, deconv_5_1, deconv_4_1, deconv_3_1, deconv_2_1, deconv_1_1]

            self.seg_score = conv(deconv_1_1, [1, 1, 64, 2], [2], bn = False, activation = False)
            self.softmax = tf.nn.softmax(self.seg_score)
            logits = tf.reshape(self.seg_score, [-1, 2])
            labels = tf.reshape(self.y, [-1, 2])
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.GradientDescentOptimizer(opts.learning_rate).minimize(self.loss)

            self.init_tf_vars = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def init(self):
        self.session.run(self.init_tf_vars)
        print('Variables initialized!')

    def train(self):
        print('Starting to train the network...')
        opts = self.options

        num_train = self.train_data.shape[0]
        for epoch in range(opts.num_epochs):
            perm_indices = np.random.permutation(range(num_train))
            average_loss = 0
            num_steps = num_train // opts.batch_size
            for step in range(num_steps):
                batch_indices = perm_indices[step * opts.batch_size:(step + 1) * opts.batch_size]
                batch_data = self.train_data[batch_indices,:,:,:]
                batch_labels = self.train_labels[batch_indices,:,:,:]
                feed_dict = {self.x: batch_data, self.y: batch_labels, self.phase: True, self.keep_prob: opts.keep_prob}
                _, loss_val = self.session.run([self.optimizer, self.loss], feed_dict = feed_dict)
                print('Epoch {:03d} step {:03d}: loss = {}'.format(epoch + 1, step + 1, loss_val), end = '\r')
                average_loss += loss_val
            print('Average loss for epoch {:03d} = {}'.format(epoch + 1, average_loss / num_steps))

            if epoch > 0 and epoch % 10 == 0:
                print('Saving trained model...')
                self.saver.save(self.session, os.path.join(opts.save_path, 'model.ckpt'))

        print('Saving trained model...')
        self.saver.save(self.session, os.path.join(opts.save_path, 'model.ckpt'))

    def restore(self):
        print('Restoring from a pre-trained model...')
        self.saver.restore(self.session, os.path.join(self.options.save_path, 'model.ckpt'))

    def predict(self, data):
        print('Generating predictions on the given data...')
        opts = self.options
        num_examples = data.shape[0]
        img_size = data.shape[1]
        predictions = np.empty((num_examples, img_size, img_size))
        for i in range(num_examples // opts.batch_size):
            s = i * opts.batch_size
            t = s + opts.batch_size
            feed_dict = {self.x: data[s:t,:,:,:], self.phase: False, self.keep_prob: 1.0}
            predictions[s:t,:,:] = self.session.run(self.softmax[:,:,:,1], feed_dict = feed_dict)
        return predictions
