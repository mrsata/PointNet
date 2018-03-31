from functools import lru_cache
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
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 10]')
parser.add_argument('--batch_size', type=int, default=50, help='Batch Size during training [default: 50]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
FLAGS = parser.parse_args()


B = FLAGS.batch_size
N = FLAGS.num_point
LR = FLAGS.learning_rate
MAX_EPOCH = FLAGS.max_epoch
LOG_DIR = FLAGS.log_dir
DISPITER = 500
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


def log(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


@lru_cache()
def load_pcloud(data, key):
    input = data[key]["points"][:]
    input = np.asarray(input, dtype=np.float32)
    label = data[key].attrs["label"]
    return input, label


def load_batch(data, b, is_training):
    keys = list(data.keys())
    if is_training:
        inputs = np.zeros((B, N, 3))
        labels = np.zeros(B)
        for i, key in enumerate(keys[b:b+B]):
            input, label = load_pcloud(data, key)
            indices = np.random.permutation(input.shape[0])
            input = input[indices[:N]]
            inputs[i] = input
            labels[i] = label
    else:
        input, label = load_pcloud(data, keys[b])
        inputs = input[np.newaxis]
        labels = label[np.newaxis]
    return inputs, labels


def train():

    data_train = h5py.File("data/3DMNIST/train_point_clouds.h5", "r")
    data_test = h5py.File("data/3DMNIST/test_point_clouds.h5", "r")

    with tf.Graph().as_default():

        inputs, labels = placeholder(B, N)
        is_training = tf.placeholder(tf.bool, shape=())
        pred = get_model(inputs, is_training)
        loss = get_loss(pred, labels)

        optimizer = tf.train.AdamOptimizer(learning_rate=LR)
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
        }

        log('\nStart training\n')
        start = time()

        for ep in range(MAX_EPOCH):

            log("#### EPOCH {:03} ####".format(ep + 1))
            begin = time()
            train_one_epoch(data_train, sess, ops)
            log("---- Time elapsed: {:.2f}s".format(time() - begin))
            eval_one_epoch(data_test, sess, ops)
            # save_path = saver.save(sess, "log/model_B%dN%d.ckpt" % (B, N//1000))
            save_path = saver.save(sess, "log/model.ckpt")

        log("Total time: {:.2f}s".format(time() - start))


def train_one_epoch(data, sess, ops):
    is_training = True
    total_corr = 0.; total_loss = 0.; total_seen = 0.

    for b in range(0, len(data), B):
        if b + B > len(data): break
        pclouds, digits = load_batch(data, b, is_training=is_training)
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
    is_training = False
    total_corr = 0; total_loss = 0; total_seen = 0

    for b in range(0, len(data), B):
        if b + B > len(data): break
        pclouds, digits = load_batch(data, b, is_training=True)
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
