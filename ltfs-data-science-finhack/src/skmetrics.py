#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 09:13:14 2021


@author: sudhir
"""

# =============================================================================
# Import libary
# =============================================================================
import os
import gc
import pandas as pd
import numpy as np
from sklearn import metrics

refresh_log = False
# =============================================================================
# Metric
# =============================================================================


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
    auc = metrics.roc_auc_score(y_valid, y_prob, multi_class="ovr")
    auc_tr = metrics.roc_auc_score(y_train, y_prob_train, multi_class="ovr")
    f1 = metrics.f1_score(y_valid, y_pred, average="macro")
    report = metrics.classification_report(y_valid, y_pred)
    con_mat = metrics.confusion_matrix(y_valid, y_pred)

    # save log
    store_result = f"""
    '{model}'
    Train dataset Absolute Accuracy\t: {trscore}% 
    Test dataset Absolute Accuracy\t: {tscore}%
    Train dataset Area Under the Curve (AUC)\t: {auc_tr}
    Test dataset Area Under the Curve (AUC)\t: {auc}
    Log Loss\t: {log_loss}
    F1 score\t: {f1}
    
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

    gc.collect()


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
        F1 score\t: {f1}

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


# custom metric
def evalerror(y_true, y_pred):
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    f_score = metrics.f1_score(y_true, y_pred, average="macro")
    return "f1_score", f_score, True


def evalerror1(preds, dtrain):
    labels = dtrain.get_label()
    preds = preds.reshape(-1, 7)
    preds = preds.argmax(axis=1)
    f_score = metrics.f1_score(preds, labels, average="macro")
    return "f1_score", f_score, True
