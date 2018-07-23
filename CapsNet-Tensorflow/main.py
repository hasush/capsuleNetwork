import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import cfg
from utils import load_data
from utils import load_multimnist
from utils import load_mnist
from utils import create_multimnist

from capsNet import CapsNet
from cnn import cnn
from cnn_multinist import cnn_multinist

#### ASDF ####
import matplotlib.pyplot as plt


def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_acc = cfg.results + '/train_acc.csv'
        val_acc = cfg.results + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return(fd_train_acc, fd_loss, fd_val_acc)
    else:
        test_acc = cfg.results + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return(fd_test_acc)


def train(model, supervisor, num_label):

    # Load the dataset.
    if not cfg.use_multimnist:
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(cfg.dataset, cfg.batch_size, is_training=True)
    else:
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_multimnist(is_training=True)
 
    # Set up objects to save data.
    fd_train_acc, fd_loss, fd_val_acc = save_to()

    # Configuration for graph.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Begin the session.
    with supervisor.managed_session(config=config) as sess:

        # Info.
        print("\nNote: all of results will be saved to directory: " + cfg.results)

        # Loop the epochs.
        for epoch in range(cfg.epoch):

            # Info.
            print("Training for epoch %d/%d:" % (epoch, cfg.epoch))

            # Check for an error that has been raised.
            if supervisor.should_stop():
                print('supervisor stoped!')
                break

            # Loop the batches.
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):

                # Obtain the indices of the data needed for training.
                start = step * cfg.batch_size
                stop = start + cfg.batch_size

                # Define the global step given the epoch and batch step.
                global_step = epoch * num_tr_batch + step

                # Iteration when we will check accuracy and record it to file in addition to training.
                if global_step % cfg.train_sum_freq == 0:

                    # Train the model and obtain information.
                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary],
                     {model.X:trX[start:stop], model.labels:trY[start:stop], model.keep_prob:cfg.keep_prob})
                    
                    # Check if the loss is NaN.
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'

                    # Add summary to the supervisor.
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    # Write the contents to disk.
                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()

                # Train the model.
                else:
                    sess.run(model.train_op, {model.X:trX[start:stop], model.labels:trY[start:stop], model.keep_prob:cfg.keep_prob})

                # Check validation set on current model. No training is done.
                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:

                    # Set the validation accuracy to zero.
                    val_acc = 0

                    # Loop over batches in the validation set.
                    for i in range(num_val_batch):

                        # Obtain the indices of the data needed for validation.
                        start = i * cfg.batch_size
                        stop = start + cfg.batch_size

                        # Compute the accuracy on the validation set.
                        acc = sess.run(model.accuracy, {model.X: valX[start:stop], model.labels: valY[start:stop]})
                        # acc, summary_str = sess.run([model.accuracy, model.train_summary_val], {model.X: valX[start:stop], model.labels: valY[start:stop]})
                        
                        # Increment the validation accuracy.
                        val_acc += acc

                    # Normalize the validation accuracy.
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)

                    # Add summary to the supervisor.
                    # supervisor.summary_writer.add_summary(summary_str, global_step)

                    # Save to file.
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            # Save the training model to file.
            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        # Close files recording information.
        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()

def evaluation(model, supervisor, num_label):

    # Load the dataset.
    if not cfg.use_multimnist:
        teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    else:
        teX, teY, num_te_batch = load_multimnist(is_training=False)

    # Object for saving results.
    fd_test_acc = save_to()

    # Start the session.
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # Restore the saved model.
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        # Initialize the test accuracy to zero.
        test_acc = 0

        # Loop over batches in the test set.
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):

            # Obtain the indices of the data needed for validation.
            start = i * cfg.batch_size
            stop = start + cfg.batch_size

            # Compute the accuracy.
            acc = sess.run(model.accuracy, {model.X: teX[start:stop], model.labels: teY[start:stop]})

            # Increment the global accuracy.
            test_acc += acc


        #### ASDF ####

        # test_acc = 0
        # for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
        #     start = i * cfg.batch_size
        #     stop = start + cfg.batch_size
        #     acc, labels, v_J, W_matrix, b_IJ, s_J, c_IJ, u_hat, biases = sess.run([model.accuracy, 
        #                                                                           model.labels,
        #                                                                           model.v_J,
        #                                                                           model.W,
        #                                                                           model.b_IJ,
        #                                                                           model.s_J,
        #                                                                           model.c_IJ,
        #                                                                           model.u_hat,
        #                                                                           model.biases],
        #                                                                           {model.X: teX[start:stop], model.labels: teY[start:stop]})
        #     test_acc += acc

        #     if i == num_te_batch - 1:
        #         np.save('saved_arrays/labels.npy', labels)
        #         np.save('saved_arrays/v_j.npy', v_J)
        #         np.save('saved_arrays/weight_matrix.npy', W_matrix)
        #         np.save('saved_arrays/b_ij.npy', b_IJ)
        #         np.save('saved_arrays/s_j.npy', s_J)
        #         np.save('saved_arrays/c_ij.npy', c_IJ)
        #         np.save('saved_arrays/u_hat.npy', u_hat)
        #         np.save('saved_arrays/biases.npy', biases)

        #### END ASDF ####
          
        # Normalize the accuracy.
        test_acc = test_acc / (cfg.batch_size * num_te_batch)

        # Save the information to disk.
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')

def print_arrays(model, supervisor):
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))

        #### ASDF ####
        print(sess.run(model.testConst))
        labels = sess.run(model.labels)
        v_J = sess.run(model.v_J)
        W_matrix = sess.run(model.W)
        b_IJ = sess.run(model.b_IJ)
        s_J = sess.run(model.s_J)
        c_IJ = sess.run(model.c_IJ)
        u_hat = sess.run(model.u_hat)
        biases = sess.run(model.biases)

        np.save('saved_arrays/labels.npy', labels)
        np.save('saved_arrays/v_j.npy', v_J)
        np.save('saved_arrays/weight_matrix.npy', W_matrix)
        np.save('saved_arrays/b_ij.npy', b_IJ)
        np.save('saved_arrays/s_j.npy', s_J)
        np.save('saved_arrays/c_ij.npy', c_IJ)
        np.save('saved_arrays/u_hat.npy', u_hat)
        np.save('saved_arrays/biases.npy', biases)

        # sys.exit()
        #### END ASDF ####


def main(_):

    # ### Create the multimnist set.
    # create_multimnist(is_training=cfg.is_training)
    # trX, trY, num_tr_batch, valX, valY, num_val_batch = load_multimnist(is_training=cfg.is_training)
    # while True:
    #     index = np.random.randint(550000)
    #     print(trY[index])
    #     plt.imshow(np.squeeze(trX[index]))
    #     plt.show()
    # sys.exit()

    # The number of labels for the data.
    num_label = 10

    # Generate the model.
    tf.logging.info('Loading Graph...')
    if cfg.use_cnn == True:
        model = cnn()
    else:
        model = CapsNet()       

    tf.logging.info('Graph loaded')

    # Generate the supervisor.
    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)
    
    # Run training or evaluation.
    if cfg.is_training:
        tf.logging.info('Start training')
        train(model, sv, num_label)
        tf.logging.info('Training done')
    else:
        tf.logging.info('Start evaluation')
        evaluation(model, sv, num_label)
        tf.logging.info('Evaluation done')

    # if cfg.is_training:
    #     print_arrays(model, sv)


if __name__ == "__main__":
    tf.app.run()