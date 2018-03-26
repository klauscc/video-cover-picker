#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: test_weights.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/03/26
#   description:
#
#================================================================

import os
import pandas as pd
import numpy as np

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

from fit_weights import readDataset, cal_score


def test_weights(X, Y, w):
    """TODO: Docstring for test_weights.

    Args:
        X (TODO): TODO
        Y (TODO): TODO
        w (TODO): TODO

    Returns: TODO

    """
    num_samples = len(Y)
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
    print(w, acc)


if __name__ == "__main__":
    csv_file_path = os.path.join(CURRENT_FILE_PATH,
                                 "./data/score_with_label.csv")
    X, Y = readDataset(csv_file_path)

    W = [
        0.81569964, 0.82407916, 0.1512219, 0.05841703, 0.07967009, 0.04343251,
        -0.04610122
    ]
    W = [
        0.04742906, 0.04793837, 0.00876862, 0.00346148, 0.00461504, 0.00253742,
        -0.0026541
    ]
    test_weights(X, Y, W)
