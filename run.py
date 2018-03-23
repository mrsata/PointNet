import h5py
import tensorflow as tf
import numpy as np

from model import placeholder, get_model, get_loss


B = 8
N = 1024 * 2
MAX_EPOCH = 10000

def load_pcloud(hf, keys):
    pclouds = np.zeros((B, N, 3))
    labels = np.zeros(B)
    for key in keys:
        pcloud = hf[key]["points"][:]
        pcloud = np.asarray(pcloud, dtype=np.float32)
        indices = np.arange(pcloud.shape[0])
        np.random.shuffle(indices)
        pcloud = pcloud[indices[:N]]
        label = hf[key].attrs["label"]
        pclouds[keys.index(key)] = pcloud
        labels[keys.index(key)] = label
    return pclouds, labels


def train():

    hf = h5py.File("input/train_point_clouds.h5", "r")
    keys = list(hf.keys())

    with tf.Graph().as_default():

        inputs, labels = placeholder(B, N)
        is_training = tf.placeholder(tf.bool, shape=())
        pred = get_model(inputs, is_training)
        loss = get_loss(pred, labels)

        optimizer = tf.train.AdamOptimizer(learning_rate=.001)
        train_op = optimizer.minimize(loss)

        tf.set_random_seed(1234)
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

        print('\nStart training\n')

        total_correct = 0
        total_loss = 0
        total_seen = 0

        for epoch in range(0, MAX_EPOCH, B):

            pclouds, digits = load_pcloud(hf, keys[epoch % len(keys):epoch % len(keys)+ B])
            feed_dict = {ops['inputs']: pclouds,
                         ops['labels']: digits,
                         ops['is_training']: True,}
            _, loss_val, pred_val = sess.run([ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            # if np.argmax(pred_val) == digits[0]:
            #     correct += 1
            # accr = correct
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == digits)
            total_correct += correct
            total_loss += loss_val * B
            total_seen += B

            if epoch % 100 == 0:
                print('epoch: {:3}, loss: {:4f}, accr: {:4f}'.format(epoch, total_loss / total_seen, total_correct / total_seen))
                # print('epoch: {:3}, loss: {:4f}, accr: {}%'.format(epoch + 1, loss_val, accr))
                # print(pred_val)


if __name__ == "__main__":
    train()
