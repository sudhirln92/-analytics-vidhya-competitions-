#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 09:13:14 2021


@author: sudhir
"""

# =============================================================================
# Import libary
# =============================================================================
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

refresh_log = False
# =============================================================================
# Metric
# =============================================================================


def eval_score(model, X_valid, y_valid):
    y_prob = model.predict_proba(X_valid)[:, 1]
    auc = metrics.roc_auc_score(y_valid, y_prob)
    return auc


def classifier_eval(model, X_train, X_valid, y_train, y_valid):
    # predict
    y_prob_train = model.predict_proba(X_train)
    y_pred_train = y_prob_train.argmax(axis=1)
    y_prob = model.predict_proba(X_valid)
    y_pred = y_prob.argmax(axis=1)

    # accuracy score
    trscore = metrics.accuracy_score(y_train, y_pred_train)
    trscore = round(trscore, 2) * 100
    tscore = metrics.accuracy_score(y_valid, y_pred)
    tscore = round(tscore, 2) * 100

    log_loss = metrics.log_loss(y_valid, y_prob)
    auc_tr = metrics.roc_auc_score(y_train, y_prob_train[:, 1])
    auc = metrics.roc_auc_score(y_valid, y_prob[:, 1])
    report = metrics.classification_report(y_valid, y_pred)
    con_mat = metrics.confusion_matrix(y_valid, y_pred)

    # save log
    store_result = f"""
    '{model}'
    Train dataset Absolute Accuracy\t: {trscore}% 
    Test dataset Absolute Accuracy\t: {tscore}%
    Train dataset ROC-AUC\t: {auc_tr}%
    Test dataset ROC-AUC\t: {auc}%
    Log Loss\t: {log_loss}
    
    Classification Report :\n {report}
    Confusion Matrix :\n {con_mat}
    """
    print(store_result)

    file_name = "model_preds/result.txt"
    if os.path.isfile(file_name) and not refresh_log:
        with open(file_name, "a") as f:
            f.write(store_result)
    else:
        with open(file_name, "w") as f:
            f.write(store_result)

    return auc


def classifier_result(y, y_pred, y_prob, X=None, model=None, printf=False):
    # predict
    if model is not None and X is not None:
        y_prob = model.predict_proba(X)
        y_pred = y_prob.argmax(axis=1)

    # accuracy score
    tscore = metrics.accuracy_score(y, y_pred)
    tscore = round(tscore, 2) * 100

    log_loss = metrics.log_loss(y, y_prob)
    auc = metrics.roc_auc_score(y, y_prob, multi_class="ovr")
    f1 = metrics.f1_score(y, y_pred, average="macro")
    report = metrics.classification_report(y, y_pred)
    con_mat = metrics.confusion_matrix(y, y_pred)

    # save log
    if printf:
        store_result = f"""
        '{model}'
        Area Under the Curve (AUC)\t: {round(auc,4)}
        Test dataset Absolute Accuracy\t: {tscore} %
        Log Loss\t: {round(log_loss,4)}

        ---------------------
        Classification Report :\n {report}
        Confusion Matrix :\n {con_mat}
        """
        print(store_result)

        file_name = "model_preds/prediction.txt"
        if os.path.isfile(file_name) and not refresh_log:
            with open(file_name, "a") as f:
                f.write(store_result)
        else:
            with open(file_name, "w") as f:
                f.write(store_result)
    return f1, con_mat


def plot_roc_cm(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)

    fig = plt.figure(figsize=(14, 6))

    fig.add_subplot(1, 2, 1)
    plt.title("Reciever Operating Charactaristics")
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")

    # confusion_matrix
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    matrix = np.array([[tp, fp], [fn, tn]])

    # plot
    fig.add_subplot(1, 2, 2)
    sns.heatmap(matrix, annot=True, cmap="viridis", fmt="g")
    plt.xticks([0.5, 1.5], labels=[1, 0])
    plt.yticks([0.5, 1.5], labels=[1, 0])
    plt.title("Confusion matrix")
    plt.xlabel("Actual label")
    plt.ylabel("Predicted label")

    precision = round(tp / (tp + fp), 3)
    recall = round(tp / (tp + fn), 3)
    print("Accuracy:", round(acc, 3))
    print("AUC :\t", round(auc, 3))
    print("Precision:", precision)
    print("Recall:\t", recall)
