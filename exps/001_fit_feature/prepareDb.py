#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: prepareDb.py
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


def generate_database(
        sample_dir,
        ground_truth_dir,
        score_file,
        score_with_label_save_file,
):
    """TODO: Docstring for .

    Args:
        sample_dir : abs dir of samples
        ground_truth_dir : groundtruth dir
        score_file : score file
        score_with_label_save_file : save path of generate database

    Returns: none

    """
    score_df = pd.read_csv(score_file)
    num_samples, num_columns = score_df.shape
    labels = np.zeros(num_samples)
    for idx, fileName in enumerate(score_df.fileName):
        ground_truth_file_path = os.path.join(ground_truth_dir, fileName)
        if os.path.exists(ground_truth_file_path):
            labels[idx] = 1

    score_with_label_df = score_df.assign(
        label=pd.Series(labels, index=score_df.index))
    score_with_label_df.to_csv(score_with_label_save_file, index=False)

if __name__ == "__main__":
    sample_dir = os.path.join(CURRENT_FILE_PATH, "../../dataset/1000db_editorSelected/samples") 
    ground_truth_dir = os.path.join(CURRENT_FILE_PATH, "../../dataset/1000db_editorSelected/groundTruth") 
    score_file = os.path.join(CURRENT_FILE_PATH, "./data/score.csv") 
    score_with_label_save_file = os.path.join(CURRENT_FILE_PATH, "./data/score_with_label.csv") 
    generate_database(sample_dir, ground_truth_dir, score_file, score_with_label_save_file) 
