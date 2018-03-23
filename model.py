import tensorflow as tf
import numpy as np


def placeholder(B, N):
    inputs = tf.placeholder(tf.float32, shape=(B, N, 3))
    labels = tf.placeholder(tf.int32, shape=(B))
    return inputs, labels

def get_model(inputs, is_training):
    '''
    Architecture:
        T-Net 1 ( MLP(64, 128, 1024) + FC(512, 256) )
        MLP 1 (64, 64)
        T-Net 2 ( MLP(64, 128, 1024) + FC(512, 256) )
        MLP 2 (64, 128, 1024)
        Max Pooling
        FC (512, 256, k)
    Input:
        B x N x 3
    '''
    k = 10
    B, N, _ = inputs.shape
    # net = tf.expand_dims(inputs, 2)
    net = tf.reshape(inputs, [B, N, 1, 3])

    # TODO: T-Net 1

    # MLP 1
    net = tf.layers.conv2d(net, 64, [1, 1], activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 64, [1, 1], activation=tf.nn.relu)

    # TODO: T-Net 2

    # MLP 2
    net = tf.layers.conv2d(net, 64, [1, 1], activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 128, [1, 1], activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 256, [1, 1], activation=tf.nn.relu)

    # Max Pooling
    net = tf.layers.max_pooling2d(net, [N, 1], [1, 1])

    # FC
    net = tf.reshape(net, [B, -1])
    net = tf.layers.dense(net, 512, activation=tf.nn.relu)
    # net = tf.layers.dropout(net, rate=0.7, training=is_training)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)
    # net = tf.layers.dropout(net, rate=0.7, training=is_training)
    net = tf.layers.dense(net, k, activation=None)

    return net

def get_loss(pred, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    loss = tf.reduce_mean(loss)
    return loss
