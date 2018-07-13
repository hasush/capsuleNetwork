import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import cfg
from utils import load_data
from utils import load_multimnist
from utils import load_mnist
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
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(cfg.dataset, cfg.batch_size, is_training=True)
    Y = valY[:num_val_batch * cfg.batch_size].reshape((-1, 1))

    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print("Training for epoch %d/%d:" % (epoch, cfg.epoch))
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)

                sys.exit()

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc = sess.run(model.accuracy, {model.X: valX[start:end], model.labels: valY[start:end]})
                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()

def train_cnn(model, supervisor, num_label):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(cfg.dataset, cfg.batch_size, is_training=True)

    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print("Training for epoch %d/%d:" % (epoch, cfg.epoch))
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                data_in = trX[start:end]
                label_in = trY[start:end]

                data_in = np.random.random((128,28,28,1))

                print("\nDataSet Shape: ", data_in.shape)
                print("DataSet Type: ", type(data_in[0][0][0][0]))
                print("Label Type: ", type(label_in[0]))
                print("Data: ", data_in[0])
                print("Len Data: ", len(data_in))
                plt.imshow(np.squeeze(data_in[0]))
                plt.show()

                if supervisor.should_stop():
                    print('supervisor stoped!')
                    break

                if global_step % cfg.train_sum_freq == 0:
                    print('Model.X: ', model.X)

                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary], {model.X: data_in, model.labels: label_in, model.keep_prob:cfg.keep_prob})
                    # _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary], {model.keep_prob:cfg.keep_prob})


                    if supervisor.should_stop():
                        print('asdf1')
                        print('supervisor stoped!')
                        break

                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op, {model.X:data_in, model.labels:label_in, model.keep_prob:cfg.keep_prob})
                    # sess.run(model.train_op, {model.keep_prob:cfg.keep_prob})


                    if supervisor.should_stop():
                        print('asdf2')
                        print('supervisor stoped!')
                        break

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc = sess.run(model.accuracy, {model.X: valX[start:end], model.labels: valY[start:end]})

                        if supervisor.should_stop():
                            print('asdf3')
                            print('supervisor stoped!')
                            break

                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()

def train_cnn_multimnist(model, supervisor, num_label):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_multimnist(cfg.is_training)
    # trX2, trY2, num_tr_batch2, valX2, valY2, num_val_batch2 = load_mnist(cfg.is_training)


    print("type trX: ", type(trX[0][0][0][0]))


    print('dim: ', trX.shape, " y: ",trY.shape)

    print("label: ", trY[34567])
    plt.imshow(np.squeeze(trX[34567]))
    plt.show()

    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print("Training for epoch %d/%d:" % (epoch, cfg.epoch))
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                if supervisor.should_stop():
                    print('supervisor stoped!')
                    break

                # print("Start : ", start, " -- stop: ", end, " -- trXshape: ", trX[start:end].shape, " -- trXtype: ", type(trX[start:end]))

                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary],
                     {model.X:trX[start:end], model.Y:trY[start:end], model.keep_prob:cfg.keep_prob})
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()

                    # pred1, pred2, pred3, pred4, y_top_k, log_top_k = sess.run([model.correct_prediction_1, model.correct_prediction_2,
                    #     model.correct_prediction_3, model.correct_prediction_4, model.Y_top_k_indices, model.logits_top_k_indices], {model.X:trX[start:end], model.Y:trY[start:end], model.keep_prob:cfg.keep_prob})
                    # plt.imshow(np.squeeze(trX[0]))
                    # plt.show()
                    # print('\nasdf1234 : ', y_top_k[0], ' ', log_top_k[0])
                    # print('pred1: {} -- pred2: {} -- pred3: {} -- pred4: {}'.format(pred1[0],pred2[0],pred3[0],pred4[0]))
                    # sys.exit()
                else:
                    sess.run(model.train_op, {model.X:trX[start:end], model.Y:trY[start:end], model.keep_prob:cfg.keep_prob})

                
               


                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc = sess.run(model.accuracy, {model.X: valX[start:end], model.Y: valY[start:end]})
                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()


def evaluation(model, supervisor, num_label):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    fd_test_acc = save_to()
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        #### ORIGINAL ####

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size

            acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
            test_acc += acc


        #### ASDF ####

        # test_acc = 0
        # for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
        #     start = i * cfg.batch_size
        #     end = start + cfg.batch_size
        #     acc, labels, v_J, W_matrix, b_IJ, s_J, c_IJ, u_hat, biases = sess.run([model.accuracy, 
        #                                                                           model.labels,
        #                                                                           model.v_J,
        #                                                                           model.W,
        #                                                                           model.b_IJ,
        #                                                                           model.s_J,
        #                                                                           model.c_IJ,
        #                                                                           model.u_hat,
        #                                                                           model.biases],
        #                                                                           {model.X: teX[start:end], model.labels: teY[start:end]})
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
          

        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')

def evaluation_cnn(model, supervisor, num_label):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    fd_test_acc = save_to()
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        #### ORIGINAL ####

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size

            acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
            test_acc += acc


        #### ASDF ####

        # test_acc = 0
        # for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
        #     start = i * cfg.batch_size
        #     end = start + cfg.batch_size
        #     acc, labels, v_J, W_matrix, b_IJ, s_J, c_IJ, u_hat, biases = sess.run([model.accuracy, 
        #                                                                           model.labels,
        #                                                                           model.v_J,
        #                                                                           model.W,
        #                                                                           model.b_IJ,
        #                                                                           model.s_J,
        #                                                                           model.c_IJ,
        #                                                                           model.u_hat,
        #                                                                           model.biases],
        #                                                                           {model.X: teX[start:end], model.labels: teY[start:end]})
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
          

        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')

def evaluation_cnn_multimnist(model, supervisor, num_label):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    fd_test_acc = save_to()
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        #### ORIGINAL ####

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size

            acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
            test_acc += acc


        #### ASDF ####

        # test_acc = 0
        # for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
        #     start = i * cfg.batch_size
        #     end = start + cfg.batch_size
        #     acc, labels, v_J, W_matrix, b_IJ, s_J, c_IJ, u_hat, biases = sess.run([model.accuracy, 
        #                                                                           model.labels,
        #                                                                           model.v_J,
        #                                                                           model.W,
        #                                                                           model.b_IJ,
        #                                                                           model.s_J,
        #                                                                           model.c_IJ,
        #                                                                           model.u_hat,
        #                                                                           model.biases],
        #                                                                           {model.X: teX[start:end], model.labels: teY[start:end]})
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
          

        test_acc = test_acc / (cfg.batch_size * num_te_batch)
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

    tf.logging.info(' Loading Graph...')
    num_label = 10

    ### TRAIN CNN USING MNIST ###
    # model = cnn()
    # tf.logging.info(' Graph loaded')
    # sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)
    # if cfg.is_training:
    #     tf.logging.info(' Start training...')
    #     train_cnn(model, sv, num_label)
    #     tf.logging.info('Training done')
    # else:
    #     evaluation_cnn(model, sv, num_label)


    # model = CapsNet()
    # model = cnn()
    print('Before instantition of model')
    model = cnn_multinist()
    print('After instantition of model')
    tf.logging.info(' Graph loaded')
    print('asdf')
    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

    if cfg.is_training:
        tf.logging.info(' Start training...')
        # train(model, sv, num_label)
        train_cnn_multimnist(model, sv, num_label)
        tf.logging.info('Training done')
    else:
        # evaluation(model, sv, num_label)
        evaluation_cnn(model, sv, num_label)

    if cfg.is_training:
        print_arrays(model, sv)


if __name__ == "__main__":
    tf.app.run()