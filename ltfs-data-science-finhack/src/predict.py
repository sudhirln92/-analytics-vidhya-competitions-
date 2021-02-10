#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 09:13:14 2021


@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
import os
import pandas as pd
import numpy as np
import joblib
from .skmetrics import classifier_result
from . import dispatcher
from .utils.file_handler import read_config

config = read_config("config.json")
seed = config["seed"]
kfold = config["kfold"]
MODELS_FOLDER = "models"
FEATURE_PIPELINE = "pipeline_X.pkl"
TARGET_PIPELINE_y = "pipeline_y.pkl"
TEST_DATA = "input/test_folds.csv"


# =============================================================================
# Predict
# =============================================================================


class MultiClassifierPredict:
    """
    Multi Classifier Predict
    """

    def __init__(
        self,
        kfold,
        n_class,
        Id,
        target_col,
        feature_pipeline_pkl,
        target_pipeline_pkl,
        **kwargs,
    ):
        self.kfold = kfold
        self.n_class = n_class
        self.Id = Id
        self.target_col = target_col
        self.feature_pipeline_pkl = feature_pipeline_pkl
        self.target_pipeline_pkl = target_pipeline_pkl
        self.models_folder = "models"

    def mean_blend(self, X, y=None):
        """
        Mean Blending
        """
        cols = X.columns
        y_prob = np.zeros((X.shape[0], self.n_class))
        for i in range(self.n_class):
            class_col = cols[cols.str.endswith(f"{i}")]
            avg = X[class_col].mean(axis=1)
            y_prob[:, i] = avg

        # label predict
        y_pred = np.argmax(y_prob, axis=1)
        if y is not None:
            print("Mean Blending final prediction", "_" * 30)
            classifier_result(y, y_pred, y_prob, printf=True)
        return y_pred, y_prob

    def predict_folds_model_avg(self, model_name, X, y=None, print_cv=False):
        # predict proba for the classifier
        y_prob = np.zeros((X.shape[0], self.n_class))
        for f in range(self.kfold):
            # read model file
            model_file = f"{self.models_folder}/{model_name}_{f}.pkl"
            clf = joblib.load(model_file)

            # predict
            prob = clf.predict(X)
            y_prob = np.add(y_prob, prob)

            # cross valid prediction
            if print_cv:
                pred = np.argmax(prob, axis=1)
                classifier_result(y, pred, prob, printf=True)

        # average prediction of models
        row_sum = np.sum(y_prob, axis=1).reshape(-1, 1)
        y_prob = np.divide(y_prob, row_sum)
        y_pred = np.argmax(y_prob, axis=1)
        if y is not None:
            print(f"Avearage Prediction For The Model {model_name}", "_" * 10)
            classifier_result(y, y_pred, y_prob, printf=True)
        return y_pred, y_prob

    def predict_stage_model(self, STAGE_MODEL, X, y=None):
        # get all models prediction
        df_probs = []
        for model_name in STAGE_MODEL:
            _, y_prob = self.predict_folds_model_avg(model_name, X, y)
            # make_prob_dataframe
            cols = [model_name + str(w) for w in range(self.n_class)]
            probs = pd.DataFrame(y_prob, columns=cols)
            df_probs.append(probs)

        # concat result
        df_probs = pd.concat(df_probs, axis=1)
        return df_probs

    def stage_2_pred(self, X, X_prob, y, stacking=False):
        # stage 2 model predict
        if stacking:
            X = pd.concat([X, X_prob], axis=1)
        y_prob, _ = self.predict_stage_model(dispatcher.MODELS_STAGE2, X, y)
        y_prob = y_prob.values
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred, y_prob

    def predict_every_model(self, X, y=None):
        # stage 1 model predict
        if len(dispatcher.MODELS.keys()) == 1:
            # single model prediction
            model_name = list(dispatcher.MODELS.keys())[0]
            y_pred, y_prob = self.predict_folds_model_avg(model_name, X, y)
        else:
            # multiple models
            df_probs = self.predict_stage_model(dispatcher.MODELS, X, y)
            # stage 2 model predict
            # y_pred,y_prob = self.stage_2_pred(X,df_probs,y,stacking=False)

            # Mean blend
            y_pred, y_prob = self.mean_blend(df_probs, y)

        return y_pred, y_prob

    def combine_result(self, y_pred, y_prob, Id):
        pipe_file_y = f"{self.models_folder}/{self.target_pipeline_pkl}"
        pipe_y = joblib.load(pipe_file_y)

        cols = pipe_y.classes_
        y_prob = pd.DataFrame(y_prob, columns=cols)

        # inverse_transform
        submission = pd.DataFrame()
        submission[self.Id] = Id
        y_pred = pipe_y.inverse_transform(y_pred)
        y_pred = pd.DataFrame(y_pred, columns=[self.target_col])
        submission = pd.concat([submission, y_pred], axis=1)
        return submission

    def final_predict(self):
        # read test data
        df_test = pd.read_csv(TEST_DATA)

        # predict_every_model
        Id = df_test[self.Id]
        try:
            X = df_test.drop([self.Id, self.target_col], axis=1)
            y = df_test[self.target_col]
        except:
            X = df_test.drop([self.Id], axis=1)
            y = None
        y_pred, y_prob = self.predict_every_model(X, y)

        submission = self.combine_result(y_pred, y_prob, Id)
        return submission

    def predict_unseen(self, df):
        # load pickle pipeline
        pipe_file_X = f"{self.models_folder}/{self.feature_pipeline_pkl}"
        pipe_X = joblib.load(pipe_file_X)

        # apply trasform
        Id = df[self.Id]
        X = pipe_X.transform(df)

        # predict_every_model
        y_pred, y_prob = self.predict_every_model(X)
        submission = self.combine_result(y_pred, y_prob, Id)
        return submission


if __name__ == "__main__":

    # predict for test data
    param = {
        "kfold": 3,
        "n_class": 7,
        "Id": "ID",
        "target_col": "Top-up Month",
        "feature_pipeline_pkl": "pipeline_X.pkl",
        "target_pipeline_pkl": "pipeline_y.pkl",
        "test_data": "input/test_folds.csv",
    }

    # for start_date,end_date
    submission = MultiClassifierPredict(**param).final_predict()
    submission.to_csv("model_preds/submission.csv", index=False)
