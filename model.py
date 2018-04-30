import tensorflow as tf
import numpy as np


def placeholder(B, N):
    inputs = tf.placeholder(tf.float32, shape=(None, None, 3))
    labels = tf.placeholder(tf.int32, shape=(None))
    return inputs, labels


def get_model(inputs, is_training, k=10, s=1024, bn_mom=None, use_tnet=False):
    '''
    Architecture:
        T-Net 1 ( MLP(64, 128, 1024) + FC(512, 256) )
        MLP 1 (64, 64)
        T-Net 2 ( MLP(64, 128, 1024) + FC(512, 256) )
        MLP 2 (64, 128, s)
        Max Pooling
        FC (512, 256, k)
    Input:
        B x N x 3
    '''
    bn = True
    net = tf.expand_dims(inputs, 2)

    # T-Net 1
    if use_tnet:
        tnet1 = conv2d(net, 64, [1, 1], bn=bn, bn_mom=bn_mom, training=is_training)
        tnet1 = conv2d(tnet1, 128, [1, 1], bn=bn, bn_mom=bn_mom, training=is_training)
        tnet1 = conv2d(tnet1, 1024, [1, 1], bn=bn, bn_mom=bn_mom, training=is_training)
        tnet1 = tf.reduce_max(tnet1, axis=1)
        tnet1 = tf.reshape(tnet1, [-1, 1024])
        tnet1 = dense(tnet1, 512, bn=bn, bn_mom=bn_mom, training=is_training)
        tnet1 = dense(tnet1, 256, bn=bn, bn_mom=bn_mom, training=is_training)
        w1 = tf.get_variable('w1', [256, 3*3],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        b1 = tf.get_variable('b1', [3*3],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        b1 += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        tnet1 = tf.matmul(tnet1, w1)
        tnet1 = tf.nn.bias_add(tnet1, b1)
        tnet1 = tf.reshape(tnet1, [-1, 3, 3])
        net = tf.matmul(tf.squeeze(net, axis=[2]), tnet1)
        net = tf.expand_dims(net, 2)

    # MLP 1
    net = conv2d(net, 64, [1, 1], bn=bn, bn_mom=bn_mom, training=is_training)
    net = conv2d(net, 64, [1, 1], bn=bn, bn_mom=bn_mom, training=is_training)

    # T-Net 2
    if use_tnet:
        tnet2 = conv2d(net, 64, [1, 1], bn=bn, bn_mom=bn_mom, training=is_training)
        tnet2 = conv2d(tnet2, 128, [1, 1], bn=bn, bn_mom=bn_mom, training=is_training)
        tnet2 = conv2d(tnet2, 1024, [1, 1], bn=bn, bn_mom=bn_mom, training=is_training)
        tnet2 = tf.reduce_max(tnet2, axis=1)
        tnet2 = tf.reshape(tnet2, [-1, 1024])
        tnet2 = dense(tnet2, 512, bn=bn, bn_mom=bn_mom, training=is_training)
        tnet2 = dense(tnet2, 256, bn=bn, bn_mom=bn_mom, training=is_training)
        w2 = tf.get_variable('w2', [256, 64*64],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        b2 = tf.get_variable('b2', [64*64],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        b2 += tf.constant(np.eye(64).flatten(), dtype=tf.float32)
        tnet2 = tf.matmul(tnet2, w2)
        tnet2 = tf.nn.bias_add(tnet2, b2)
        tnet2 = tf.reshape(tnet2, [-1, 64, 64])
        net = tf.matmul(tf.squeeze(net, axis=[2]), tnet2)
        net = tf.expand_dims(net, 2)

    # MLP 2
    net = conv2d(net, 64, [1, 1], bn=bn, bn_mom=bn_mom, training=is_training)
    net = conv2d(net, 128, [1, 1], bn=bn, bn_mom=bn_mom, training=is_training)
    net = conv2d(net, s, [1, 1], bn=bn, bn_mom=bn_mom, training=is_training)

    # Max Pooling
    net = tf.reduce_max(net, axis=1)

    # FC
    net = tf.reshape(net, [-1, s])
    net = dense(net, 512, bn=bn, bn_mom=bn_mom, training=is_training)
    net = dense(net, 256, bn=bn, bn_mom=bn_mom, training=is_training)
    net = tf.layers.dropout(net, rate=0.7, training=is_training)
    net = dense(net, k, bn=False, activation=None)

    return net


def get_loss(pred, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    loss = tf.reduce_mean(loss)
    return loss


def conv2d(inputs, filter, kernal, bn=False, bn_mom=None, activation=tf.nn.relu, training=None):
    output = tf.layers.conv2d(inputs, filter, kernal)
    if bn:
        output = batch_norm(output, bn_mom, training)
    if activation is not None:
        output = activation(output)
    return output


def dense(inputs, units, bn=False, bn_mom=None, activation=tf.nn.relu, training=None):
    output = tf.layers.dense(inputs, units)
    if bn:
        output = batch_norm(output, bn_mom, training)
    if activation is not None:
        output = activation(output)
    return output


def batch_norm(inputs, bn_mom=None, training=None):
    if bn_mom is None: bn_mom = 0.9
    return tf.layers.batch_normalization(inputs, momentum=bn_mom, training=training)
