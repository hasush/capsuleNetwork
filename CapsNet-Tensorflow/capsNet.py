"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import tensorflow as tf

from config import cfg
from utils import get_batch_data
from utils import get_batch_data_multimnist
from utils import softmax
from utils import reduce_sum
from capsLayer import CapsLayer


epsilon = 1e-9

class CapsNet(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
     
            if not cfg.use_multimnist:
                self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)
                self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)
            else:
                self.X, self.labels = get_batch_data_multimnist(cfg.batch_size, cfg.num_threads)
                self.Y = tf.cast(self.labels, dtype=tf.float32)

            self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

            self.build_arch()
            self.loss()
            self._summary()

            # t_vars = tf.trainable_variables()
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)  # var_list=t_vars)
            

        tf.logging.info('Seting up the main structure')

    def build_arch(self):

        with tf.variable_scope('Test'):
            self.testConst= tf.constant(1.0, name='testConst')

        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            print('shape of self x : ', self.X.shape)
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=cfg.image_size-19, stride=1,
                                             padding='VALID')
            print('shape asdf asdf: ', conv1.get_shape())
            assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

        # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
            assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)

            self.v_J = digitCaps.v_J
            self.W = digitCaps.W
            self.b_IJ = digitCaps.b_IJ
            self.s_J = digitCaps.s_J
            self.c_IJ = digitCaps.c_IJ
            self.u_hat = digitCaps.u_hat
            self.biases = digitCaps.biases
            

        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2),
                                               axis=2, keepdims=True) + epsilon)
            self.softmax_v = softmax(self.v_length, axis=1)
            assert self.softmax_v.get_shape() == [cfg.batch_size, 10, 1, 1]

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size, ))

            # If using multimnist dataset, determine the indices of the vectors which have the
            # greatest activation.
            if cfg.use_multimnist:
                self.softmax_v_flatten = tf.reshape(self.softmax_v, shape=(cfg.batch_size, 10))

                # Obtain the top 2 logit values and their indices for the logits and the labels.
                self.softmax_v_top_k_values, self.softmax_v_top_k_indices = tf.nn.top_k(self.softmax_v_flatten, k=2, sorted=False)
                self.Y_top_k_values, self.Y_top_k_indices = tf.nn.top_k(self.Y, k=2, sorted=False)

                # See if one of the indices of the logits matches the labels.
                self.correct_prediction_1 = tf.equal(self.Y_top_k_indices[:, 0], self.softmax_v_top_k_indices[:, 0])
                self.correct_prediction_2 = tf.equal(self.Y_top_k_indices[:, 0], self.softmax_v_top_k_indices[:, 1])
                self.correct_prediction_3 = tf.equal(self.Y_top_k_indices[:, 1], self.softmax_v_top_k_indices[:, 0])
                self.correct_prediction_4 = tf.equal(self.Y_top_k_indices[:, 1], self.softmax_v_top_k_indices[:, 1])            

            # Method 1.
            if not cfg.mask_with_y:
                # c). indexing
                # It's not easy to understand the indexing process with argmax_idx
                # as we are 3-dim animal
                masked_v = []
                for batch_size in range(cfg.batch_size):
                    v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                self.masked_v = tf.concat(masked_v, axis=0)
                assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]
            # Method 2. masking with true label, default mode
            else:
                # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)))
                self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            assert fc1.get_shape() == [cfg.batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            assert fc2.get_shape() == [cfg.batch_size, 1024]
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=cfg.image_size_flatten, activation_fn=tf.sigmoid)

    def loss(self):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        assert max_l.get_shape() == [cfg.batch_size, 10, 1, 1]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*height*width=0.392 (if height=28)
        self.total_loss = self.margin_loss + 0.0005 * cfg.image_size_flatten * self.reconstruction_err

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, cfg.image_size, cfg.image_size, 1))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        if not cfg.use_multimnist:
            correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
            self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        else:
            # Obtain the accuracy by summing the correct matches. Note that this definition gives a maximum
            # accuracy of 2.0, thus one would need to divide the accuracy by 2 in order to obtain the true accuracy.
            self.accuracy = (tf.reduce_sum(tf.cast(self.correct_prediction_1, tf.float32))  \
                            + tf.reduce_sum(tf.cast(self.correct_prediction_2, tf.float32)) \
                            + tf.reduce_sum(tf.cast(self.correct_prediction_3, tf.float32)) \
                            + tf.reduce_sum(tf.cast(self.correct_prediction_4, tf.float32)))/2.0

        
