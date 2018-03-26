#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: fit_pairs.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/03/23
#   description:
#
#================================================================

import os
import numpy as np
import tensorflow as tf

from fit_weights import readDataset

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def prepareDatabase(csv_file_path):
    """

    Args:
        csv_file_path (TODO): TODO

    Returns:
        X: tuple. each element is a pair with bad image and good image
        Y: tuple. dummpy y. has the same size of Y

    """
    X, Y = readDataset(csv_file_path)
    X_pair = []
    for video, y in zip(X, Y):
        for idx, x in enumerate(video):
            if idx != y:
                temp_x = np.concatenate((x, video[y]))
                X_pair.append(temp_x)
    X_pair = np.array(X_pair)
    return X_pair


learning_rate = 0.01
training_epochs = 50
batch_size = 128
display_step = 1
beta = 0.1


def my_model(features):
    """TODO: Docstring for build_model.
    Returns: TODO

    """
    # features with shape (None, 14). the first 7 is bad image, the second 7 is good image
    feature_length = 7
    W = tf.Variable(
        tf.truncated_normal(
            shape=[feature_length], stddev=np.sqrt(2 / feature_length)),
        name="w")
    W1 = tf.tile(W, [2])
    multi = tf.multiply(features, W1)
    reshapeMulti = tf.reshape(multi, [-1, 2, feature_length])
    scores = tf.reduce_sum(reshapeMulti, 2)  # None x 2
    scores = tf.gather(
        scores, 0, axis=1) - tf.gather(
            scores, 1, axis=1)  #None x 1
    scores = tf.squeeze(scores)  #None
    reg = tf.nn.l2_loss(W)
    loss = tf.reduce_mean(scores + reg * beta)
    return loss, scores


def train(X_train, X_test):
    """TODO: Docstring for train.

    Args:
        X_train (TODO): TODO
        Y_train (TODO): TODO
        X_test (TODO): TODO
        Y_test (TODO): TODO

    Returns: TODO

    """
    features = tf.placeholder(tf.float32, [None, 14])
    loss, pred = my_model(features)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(X_train.shape[0] / batch_size)

            # shuffle each epoch
            p = np.random.permutation(len(X_train))
            X_train = X_train[p]
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = X_train[i * batch_size:(i + 1) * batch_size]
                _, c = sess.run(
                    [optimizer, loss], feed_dict={
                        features: batch_xs,
                    })
                avg_cost += c / total_batch
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=",
                      "{:.9f}".format(avg_cost))

            #prediction
            correct_prediction = tf.less(pred, 0)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy:", accuracy.eval({features: X_test}))
            w = [var for var in tf.global_variables() if var.op.name=="w"][0]
            print w.eval() 
        print("Optimization Finished!")

        # test_on dataset


if __name__ == "__main__":
    csv_file_path = os.path.join(CURRENT_FILE_PATH,
                                 "./data/score_with_label.csv")
    X = prepareDatabase(csv_file_path)
    Y = np.zeros(X.shape[0])
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42)
    train(X_train, X_test)
