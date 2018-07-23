import tensorflow as tf

from config import cfg
from utils import get_batch_data
from utils import get_batch_data_multimnist
from utils import softmax
from utils import reduce_sum

class cnn(object):
	def __init__(self, is_training=True):
		self.graph = tf.Graph()
		with self.graph.as_default():
			
			if not cfg.use_multimnist:
				self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)
				self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32, name='Y')
			else:
				self.X, self.labels = get_batch_data_multimnist(cfg.batch_size, cfg.num_threads)
				self.Y = self.labels

			self.keep_prob = tf.placeholder_with_default(1.0, shape=())

			self.build_arch()
			self.loss()
			self._summary()

			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.optimizer = tf.train.AdamOptimizer()
			self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

		tf.logging.info('Setting up the main structure')

	def build_arch(self):

		with tf.variable_scope('Conv1_layer'):
			self.conv1 = tf.contrib.layers.conv2d(self.X, 
											 num_outputs = 512, 
											 kernel_size = cfg.image_size-19, # Ensures that the input to the next layer is 20x20
											 stride = 1, 
											 padding = 'VALID')

			self.pool1 = tf.layers.max_pooling2d(self.conv1, 
											pool_size = [2,2],
											strides = [2,2])

		with tf.variable_scope('Conv2_layer'):
			self.conv2 = tf.contrib.layers.conv2d(self.pool1, 
											 num_outputs = 256, 
											 kernel_size = 5, 
											 stride = 1, 
											 padding = 'VALID')

			self.pool2 = tf.layers.max_pooling2d(self.conv2, 
											pool_size = [2,2],
											strides = [2,2])

		with tf.variable_scope('FC_layer'):
			self.pool2_flat = tf.reshape(self.pool2, [cfg.batch_size, -1])
			self.dense = tf.layers.dense(inputs=self.pool2_flat, units=1024, activation=tf.nn.relu)
			self.dropout = tf.nn.dropout(self.dense, self.keep_prob)

		with tf.variable_scope('logits'):
			self.logits = tf.layers.dense(inputs=self.dropout, units=10)

	def loss(self):
		self.total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

	def _summary(self):
		train_summary = []
		train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
		self.train_summary = tf.summary.merge(train_summary)

		if not cfg.use_multimnist:
			correct_prediction =  tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
			self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
		else:

			# Obtain the top 2 logit values and their indices for the logits and the labels.
			self.logits_top_k_values, self.logits_top_k_indices = tf.nn.top_k(self.logits, k=2, sorted=False)
			self.Y_top_k_values, self.Y_top_k_indices = tf.nn.top_k(self.Y, k=2, sorted=False)

			# See if one of the indices of the logits matches the labels.
			self.correct_prediction_1 = tf.equal(self.Y_top_k_indices[:, 0], self.logits_top_k_indices[:, 0])
			self.correct_prediction_2 = tf.equal(self.Y_top_k_indices[:, 0], self.logits_top_k_indices[:, 1])
			self.correct_prediction_3 = tf.equal(self.Y_top_k_indices[:, 1], self.logits_top_k_indices[:, 0])
			self.correct_prediction_4 = tf.equal(self.Y_top_k_indices[:, 1], self.logits_top_k_indices[:, 1])

			# Obtain the accuracy by summing the correct matches. Note that this definition gives a maximum
			# accuracy of 2.0, thus one would need to divide the accuracy by 2 in order to obtain the true accuracy.
			self.accuracy = (tf.reduce_sum(tf.cast(self.correct_prediction_1, tf.float32))  \
							+ tf.reduce_sum(tf.cast(self.correct_prediction_2, tf.float32)) \
							+ tf.reduce_sum(tf.cast(self.correct_prediction_3, tf.float32)) \
							+ tf.reduce_sum(tf.cast(self.correct_prediction_4, tf.float32)))/2.0