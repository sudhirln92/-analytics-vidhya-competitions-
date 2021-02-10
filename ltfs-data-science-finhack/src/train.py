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
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

from .utils.logger import logging_time
from .skmetrics import classifier_eval, evalerror
from . import dispatcher

from .utils.file_handler import read_config

config = read_config("config.json")
seed = config["seed"]
kfold = config["kfold"]

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
# FOLD = int(os.environ.get('FOLD'))
MODEL = os.environ.get("MODEL")
TARGET_COL = os.environ.get("TARGET_COL")
STAGE = int(os.environ.get("STAGE"))
SAVE_VALID = bool(os.environ.get("SAVE_VALID"))


# =============================================================================
# train
# =============================================================================


class TrainModel:
    def __init__(
        self,
        Id,
        kfold,
        drop_cols,
        target_col,
        MODEL,
        TRAINING_DATA,
        STAGE,
        SAVE_VALID=True,
        **kwargs,
    ):
        self.Id = Id
        self.kfold = kfold
        self.drop_cols = drop_cols
        self.kflod_col = "kfold"
        self.target_col = target_col
        self.MODEL = MODEL
        self.TRAINING_DATA = TRAINING_DATA
        self.STAGE = STAGE
        self.SAVE_VALID = SAVE_VALID

        self.valid_pred = []

    def get_model(self):
        """
        Get model class from dispatcher
        """
        if self.STAGE == 1:
            model = dispatcher.MODELS[self.MODEL]
        else:
            model = dispatcher.MODELS_STAGE2[self.MODEL]
        return model

    def read_train_data(self):
        # data set
        if self.STAGE == 1:
            train = pd.read_csv(self.TRAINING_DATA)
            # train = train[:50000]
            # print("training data is reduced")
        else:
            m_keys = dispatcher.MODELS.keys()
            len_key = len(m_keys)
            preds = pd.DataFrame()
            for i, k in enumerate(m_keys):
                pred = pd.read_csv(f"model_preds/{k}_pred.csv")
                print(i, k)
                columns = [self.Id, self.kflod_col]
                if i == 0:
                    preds = pred
                else:
                    preds = pd.merge(preds, pred, on=columns, how="left")

            train = pd.read_csv(self.TRAINING_DATA)
            drop_cols = []
            for w in self.drop_cols:
                if w not in [self.Id, self.kflod_col]:
                    drop_cols.append(w)
            train = train.drop(drop_cols, axis=1)
            train = pd.merge(train, preds, on=columns, how="left")
            print(train.head())

        # df_test = pd.read_csv(TEST_DATA)
        return train

    def get_kfold_ids(self, train):
        # kflod

        train_idx = train[train[self.kflod_col] != self.fold].index
        valid_idx = train[train[self.kflod_col] == self.fold].index

        return train_idx, valid_idx

    def data_split(self, train, train_idx, valid_idx):

        # split data based on kfold id
        X = train.drop(self.drop_cols, axis=1)
        y = train[[self.target_col]]
        X_train = X.loc[train_idx].reset_index(drop=True)
        X_valid = X.loc[valid_idx].reset_index(drop=True)

        y_train = y.loc[train_idx][self.target_col].values
        y_valid = y.loc[valid_idx][self.target_col].values

        return X_train, X_valid, y_train, y_valid

    @logging_time
    def train_model(self, X_train, X_valid, y_train, y_valid, save_model=True):
        # train machine learning model
        model = self.get_model()
        if self.MODEL == "lgbm":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric=evalerror,
                verbose=100,
                early_stopping_rounds=20,
            )
        elif self.MODEL == "xgbm":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="multi_error",
                verbose=100,
            )

        else:
            model.fit(X_train, y_train)

        # save model
        if save_model:
            joblib.dump(model, f"models/{self.MODEL}_{self.fold}.pkl")
        return model

    def save_valid_predict(self, model, train, X_valid, valid_idx):
        # predict
        y_prob = model.predict_proba(X_valid)
        cols = [self.target_col + "_pred" + str(w) for w in range(7)]
        y_prob = pd.DataFrame(y_prob, columns=cols)

        # save_validation prediction
        columns = [self.Id, self.kflod_col, self.target_col]
        df_valid = train.loc[valid_idx, columns].reset_index(drop=True)
        df_valid = pd.concat([df_valid, y_prob], axis=1)

        # save valid pred
        self.valid_pred.append(df_valid)

        if self.kfold == self.fold + 1:
            # save prediction
            self.valid_pred = pd.concat(self.valid_pred, axis=0).reset_index(drop=True)
            self.valid_pred.to_csv(f"model_preds/{self.MODEL}_pred.csv", index=False)

    @logging_time
    def run_training(self):

        print(f"Training {self.MODEL} model")
        # read train data
        train = self.read_train_data()

        # train model
        for self.fold in range(self.kfold):

            # get_kfold_ids
            train_idx, valid_idx = self.get_kfold_ids(train)

            # training model
            X_train, X_valid, y_train, y_valid = self.data_split(
                train, train_idx, valid_idx
            )
            model = self.train_model(X_train, X_valid, y_train, y_valid)

            classifier_eval(model, X_train, X_valid, y_train, y_valid)

            # save valid_data
            if self.SAVE_VALID:
                self.save_valid_predict(model, train, X_valid, valid_idx)


if __name__ == "__main__":

    # parameters
    param = {
        "Id": "ID",
        "kfold": kfold,
        "drop_cols": ["ID", "kfold", "Top-up Month"],
        "target_col": TARGET_COL,
        "MODEL": MODEL,
        "TRAINING_DATA": TRAINING_DATA,
        "STAGE": STAGE,
        "SAVE_VALID": SAVE_VALID,
    }

    TrainModel(**param).run_training()
