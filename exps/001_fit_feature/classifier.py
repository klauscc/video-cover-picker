#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: classifier.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/03/23
#   description:
#
#================================================================
import os
import pandas as pd
import numpy as np
import sklearn

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

RANDOM_STATE = 42

names = [
    "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Neural Net",
    "AdaBoost", "Naive Bayes", "QDA"
]
classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()
]


def readDataset(csv_file_path):
    """TODO: Docstring for readDataset.

    Args:
        csv_file_path (TODO): TODO

    Returns: TODO

    """
    data_pd = pd.read_csv(csv_file_path)
    X = data_pd.iloc[:, 1:10]
    y = data_pd.iloc[:, 10]
    X = np.array(X) 
    y = np.array(y) 

    # nan to zero
    where_are_NaNs = np.isnan(X)
    X[where_are_NaNs] = 0
    return X, y


def train(X, y):
    """TODO: Docstring for train.
    Returns: TODO

    """
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    for (name, clf) in zip(names, classifiers):
        print("=========training {}...==========".format(name))
        pipe = make_pipeline_imb(NearMiss(), clf)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print(classification_report_imbalanced(y_test, y_pred))


if __name__ == "__main__":
    csv_file_path = os.path.join(CURRENT_FILE_PATH,
                                 "./data/score_with_label.csv")
    X, y = readDataset(csv_file_path)
    train(X, y)
