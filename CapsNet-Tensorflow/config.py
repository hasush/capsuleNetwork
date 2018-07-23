import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 27, 'batch size')
flags.DEFINE_integer('epoch', 50, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')


############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'mnist', 'The name of dataset [mnist, fashion-mnist')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')
flags.DEFINE_string('logdir', 'archived_results/capsules_percent_data_epoch_constant_test/capsules_percent_data_0p1_2batch/logdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 100, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 500, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')
flags.DEFINE_string('results', 'archived_results/capsules_percent_data_epoch_constant_test/capsules_percent_data_0p1_2batch/results', 'path for saving results')

############################
#   distributed setting    #
############################
flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 27, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

##################
#   misc test    #
##################
flags.DEFINE_boolean('use_cnn', False, 'if True then use a cnn, if False use a capsule network') #XXXX

flags.DEFINE_boolean('use_multimnist', False, 'whether to use the multimnist data set') #XXXX
flags.DEFINE_boolean('use_affnist_train', False, 'whether to use the affnist data set for training')
flags.DEFINE_boolean('use_affnist_test', False, 'whether to use the affnist data set for testing')
flags.DEFINE_boolean('use_10_images', False, 'whether to use only a ten image data set for training')

flags.DEFINE_boolean('use_image_transformation_train_pad_zero', False, 'whether to modify images by padding with zeros in the train phase') #XXX
flags.DEFINE_boolean('use_image_transformation_train_rotation', False, 'whether to modify images by rotating in the train phase') #XXX
flags.DEFINE_boolean('use_image_transformation_train_translation', False, 'whether to modify images by translating in the train phase') #XXX
flags.DEFINE_float('percent_train_use', 0.001, 'number in (0,1], if 1, then use all of the data') #XXX

flags.DEFINE_boolean('use_image_transformation_test_pad_zero', False, 'whether to modify images by padding with zeros in the train phase') #XXX
flags.DEFINE_boolean('use_image_transformation_test_rotation', False, 'whether to modify images by rotating in the train phase') #XXX
flags.DEFINE_boolean('use_image_transformation_test_translation', False, 'whether to modify images by translating in the train phase') #XXX

flags.DEFINE_integer('max_rotation_angle', 0, 'angle in degrees to which the image can be rotated by')
flags.DEFINE_integer('max_translation_pixel', 5, 'number of pixels to translate image by')
flags.DEFINE_integer('num_sampled_images_needed_multimnist', 10, 'number of images to sample per mnist image for creation of the multimnist data set')

flags.DEFINE_integer('image_size', 28, 'width and height of the image')
flags.DEFINE_integer('image_size_flatten', 784, 'width*height of the image')

flags.DEFINE_string('affNistTest', '/home/gsandh16/Documents/capsuleNetwork/CapsNet-Tensorflow/data/affNist/test.mat', 'the path to the affnist test set')
flags.DEFINE_string('multimnist_path', '/home/gsandh16/Documents/capsuleNetwork/CapsNet-Tensorflow/data/multi_mnist/', 'the path to the affnist test set')

flags.DEFINE_float('keep_prob', 0.5, 'the keep probability of the dropout layers attached the cnn fully connected layer')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
