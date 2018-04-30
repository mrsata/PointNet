from time import time
import argparse
import os
import h5py
import tensorflow as tf
import numpy as np
np.random.seed(0)
tf.set_random_seed(0)

from model import placeholder, get_model, get_loss


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('-t', '--tnet', type=bool, default=False, help='Whether use T-Net [default: False]')
parser.add_argument('-n', '--num_point', type=int, default=1024, help='Point Number [64/128/512/1024/2048] [default: 1024]')
parser.add_argument('-e', '--num_point_eval', type=int, default=1024, help='Point Number for Evaluation [64/128/512/1024/2048] [default: 1024]')
parser.add_argument('-s', '--size_of_layer', type=int, default=1024, help='Size of the last layer [512/1024/2048] [default: 1024]')
parser.add_argument('-m', '--max_epoch', type=int, default=25, help='Epoch to run [default: 25]')
parser.add_argument('-b', '--batch_size', type=int, default=50, help='Batch Size during training [default: 50]')
parser.add_argument('-bm', '--bn_mom', type=float, default=0.9, help='Batch normalization momentum [default: 0.9]')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
FLAGS = parser.parse_args()


B = FLAGS.batch_size
S = FLAGS.size_of_layer
N = FLAGS.num_point
NE = FLAGS.num_point_eval
LR = FLAGS.learning_rate
MAX_EPOCH = FLAGS.max_epoch
LOG_DIR = FLAGS.log_dir
TNET = FLAGS.tnet
BN_MOM = FLAGS.bn_mom
DISPITER = 500//B*B if 500-500//B*B < 500//B*B+B-500 else 500//B*B+B
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, LOG_DIR + '.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


def log(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def load_pcloud(data, key):
    input = data[key]["points"][:]
    input = np.asarray(input, dtype=np.float32)
    label = data[key].attrs["label"]
    return input, label


def load_batch(data, b):
    inputs, labels = data
    return inputs[b:b+B], labels[b:b+B]


def load_data(data, num_point):
    keys = list(data.keys())
    inputs = np.zeros((len(data), num_point, 3))
    labels = np.zeros(len(data))
    for i, key in enumerate(keys):
        input, label = load_pcloud(data, key)
        indices = np.random.permutation(input.shape[0])
        input = input[indices[:num_point]]
        inputs[i] = input
        labels[i] = label
    return inputs, labels


def train():

    start = time()
    file_train = h5py.File("data/3DMNIST/train_point_clouds.h5", "r")
    file_test = h5py.File("data/3DMNIST/test_point_clouds.h5", "r")
    data_train = load_data(file_train, N)
    data_test = load_data(file_test, NE)
    log('Data loaded in %2fs' % (time() - start))

    with tf.Graph().as_default():

        is_training = tf.placeholder(tf.bool, shape=())
        inputs, labels = placeholder(B, N)
        pred = get_model(inputs, is_training, k=10, s=S, use_tnet=TNET, bn_mom=BN_MOM)
        loss = get_loss(pred, labels)

        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        ops = {
            'inputs': inputs,
            'labels': labels,
            'is_training': is_training,
            'pred': pred,
            'loss': loss,
            'train_op': train_op,
            'learning_rate': LR,
        }

        log('\nStart training\n')
        start = time()

        for ep in range(MAX_EPOCH):
            if ep == 20: ops['learning_rate'] /= 10
            log("#### EPOCH {:03} ####".format(ep + 1))
            begin = time()
            train_one_epoch(data_train, sess, ops)
            log("---- Time elapsed: {:.2f}s".format(time() - begin))
            eval_one_epoch(data_test, sess, ops)
            # save_path = saver.save(sess, "log/model_B%dN%d.ckpt" % (B, N//1000))
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))

        log("Total time: {:.2f}s".format(time() - start))


def train_one_epoch(data, sess, ops):
    is_training = True
    total_corr = 0.; total_loss = 0.; total_seen = 0.

    for b in range(0, len(data[0]), B):
        pclouds, digits = load_batch(data, b)
        feed_dict = {ops['inputs']: pclouds,
                     ops['labels']: digits,
                     ops['is_training']: is_training,}
        _, loss_val, pred_val = sess.run([ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == digits)
        total_corr += correct
        total_loss += loss_val * B
        total_seen += B

        if (b+B) % DISPITER == 0:
            loss = total_loss / total_seen
            accr = total_corr / total_seen * 100
            log('{:04} loss: {:.4f} accr: {:.2f}%'.format(b+B, loss, accr))
            total_corr = 0.; total_loss = 0.; total_seen = 0.


def eval_one_epoch(data, sess, ops):
    is_training = True
    total_corr = 0; total_loss = 0; total_seen = 0

    for b in range(0, len(data[0]), B):
        pclouds, digits = load_batch(data, b)
        feed_dict = {ops['inputs']: pclouds,
                     ops['labels']: digits,
                     ops['is_training']: is_training,}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)

        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == digits)
        total_corr += correct
        total_loss += loss_val * B
        total_seen += B

    loss = total_loss / total_seen
    accr = total_corr / total_seen * 100
    log('Eval loss: {:.4f} accr: {:.2f}%'.format(loss, accr))


if __name__ == "__main__":
    train()
