#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: fit_weights.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/03/23
#   description:
#
#================================================================

import os
import pandas as pd
import numpy as np

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def readDataset(csv_file_path):
    """TODO: Docstring for readDataset.

    Args:
        csv_file_path (TODO): TODO

    Returns: TODO

    """
    data_pd = pd.read_csv(csv_file_path)
    data_pd.fillna(0)
    datas = np.array(data_pd)

    X = []
    Y = []

    current_video_name = None
    x = []
    y = -1
    video_image_index = 0
    for data in datas:
        filename = data[0]
        features = np.array(data[[2, 3, 4, 6, 7, 8, 9]]).astype(np.float)
        label = data[10]
        video_name = filename.split("/")[0]

        if video_name != current_video_name:
            if y != -1:
                x = np.array(x).astype(np.float)
                X.append(x)
                Y.append(y)
            x = []
            y = -1
            video_image_index = 0
        x.append(features)
        if label == 1:
            y = video_image_index
        video_image_index += 1

        current_video_name = video_name

    # Y = np.asarray(Y, dtype=np.float)

    return X, Y


def cal_score(w, x):
    """TODO: Docstring for cal_score.

    Args:
        w :

    Returns: TODO

    """
    return np.sum(np.multiply(w, x))


def fit_weights(X, Y):
    """TODO: Docstring for fit_weights.

    Args:
        X (TODO): list. each element is a 2d-array
        Y (TODO): TODO

    Returns: TODO

    """

    num_samples = len(Y)
    ar = np.arange(0, 3, 0.3)
    max_accuracy = 0
    best_w = None
    for w6 in ar:
        for w4 in ar:
            for w3 in ar:
                for w5 in ar:
                    for w2 in ar:
                        for w1 in ar:
                            for w0 in ar:
                                w = (w0, w1, w2, w3, w4, w5, w6)
                                num_correct = 0
                                for video, y in zip(X, Y):
                                    max_score = 0
                                    max_y = 0
                                    for idx, x in enumerate(video):
                                        current_score = cal_score(w, x)
                                        if current_score > max_score:
                                            max_score = current_score
                                            max_y = idx
                                    if max_y == y:
                                        num_correct += 1
                                acc = float(num_correct) / num_samples
                                if acc > max_accuracy:
                                    max_accuracy = acc
                                    best_w = w
                                    print(best_w, max_accuracy)


if __name__ == "__main__":
    csv_file_path = os.path.join(CURRENT_FILE_PATH,
                                 "./data/score_with_label.csv")
    X, y = readDataset(csv_file_path)
    print len(X)
    print len(y)
    fit_weights(X, y)
