#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 09:13:14 2021

@author: sudhir
"""

# =============================================================================
# Import library
# =============================================================================
import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import ensemble
from sklearn import neighbors
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from functools import partial
import lightgbm as lgb
import xgboost as xgb
import argparse

from .utils.logger import logging_time
from .skmetrics import eval_score
from . import dispatcher


seed = 42
# =============================================================================
# argument
# =============================================================================

my_parser = argparse.ArgumentParser(allow_abbrev=True)
my_parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="""Model name log_reg, sgd, rftree, extree, lgbm, xgbm """,
)
my_parser.add_argument(
    "--refresh_log",
    type=bool,
    default=False,
    help="""Refresh optimal parameter log file will erase past parameter result""",
)
my_parser.add_argument(
    "--target_col",
    required=True,
    type=str,
    default=False,
    help="""target_col: Top-up Month """,
)

# my_parser.add_argument('train', help=""" path : input/train_folds.csv""")
refresh_log = my_parser.parse_args().refresh_log
MODEL = my_parser.parse_args().model
TARGET_COL = my_parser.parse_args().target_col
TRAINING_DATA = "input/train_folds.csv"
# TRAINING_DATA = os.environ.get("TRAINING_DATA")
# MODEL = os.environ.get('MODEL')
# =============================================================================
# optimize
# =============================================================================


def optimize(params, X, y):
    """
    The optimization function
    """
    print(params)
    # Choice model
    global MODEL
    if MODEL == "log_reg":
        model = linear_model.LogisticRegression(
            **params, multi_class="multinomial", random_state=seed
        )
    elif MODEL == "lgbm":
        model = lgb.LGBMClassifier(
            **params,
            learning_rate=0.01,
            n_estimators=2000,
            objective="binary",
            boosting_type="gbdt",
            random_state=seed,
        )

    # initialise stratified k-fold
    kf = model_selection.StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)
    all_score = []
    for (train_idx, valid_idx) in kf.split(X=X, y=y):
        # print('Fold:',i+1)
        X_train, y_train = X.loc[train_idx], y[train_idx]
        X_valid, y_valid = X.loc[valid_idx], y[valid_idx]

        # fit model
        if MODEL == "lgbm":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="auc",
                verbose=25,
                early_stopping_rounds=50,
            )
        else:
            model.fit(X_train, y_train)
        # predict
        score = eval_score(model, X_valid, y_valid)
        print("score", score)
        all_score.append(score)

    print("accuracies:\t", all_score)
    return -np.mean(all_score)


@logging_time
def BayesSearch(X, y):
    """Search Hyper parameter"""
    global MODEL
    if MODEL == "log_reg":
        param_space = {
            "solver": hp.choice("solver", ["newton-cg", "saga", "lbfgs"]),
            "max_iter": scope.int(hp.uniform("max_iter", 100, 1500)),
            "C": scope.float(hp.lognormal("C", 0.0001, 3)),
        }

    elif MODEL == "lgbm":
        param_space = {
            "max_depth": scope.int(hp.quniform("max_depth", 6, 50, 1)),
            "subsample": scope.float(hp.uniform("subsample", 0.4, 1)),
            "colsample_bytree": scope.float(hp.uniform("colsample_bytree", 0.4, 1)),
            # "subsample_freq": scope.int(hp.uniform("subsample_freq", 0, 5, 1)),
            "min_split_gain": scope.float(hp.uniform("min_split_gain", 0.01, 1)),
            "reg_alpha": scope.float(hp.uniform("reg_alpha", 0.01, 2)),
            "reg_lambda": scope.float(hp.uniform("reg_lambda", 0.01, 2)),
            "num_leaves": scope.int(hp.quniform("num_leaves", 31, 100, 2)),
            # "scale_pos_weight": scope.float(hp.uniform("scale_pos_weight", 0.001, 5)),
            # "min_child_samples": scope.int(hp.quniform("min_child_samples", 20, 200, 1)),
        }
    # optimize function
    trails = Trials()
    optimization_function = partial(optimize, X=X, y=y)
    result = fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trails,
        verbose=1,
    )

    print("Best Result is:", "_" * 10, result)
    return result, trails


def track_result(trails, MODEL, refresh_log):
    store_result = f""" 
    MODEL NAME: {MODEL},
    Trails losses: {trails.losses()},
    Trails Best result: {trails.best_trial['result']},
    Trails Time of execution: {trails.best_trial['book_time']},
    Parameters: {trails.argmin}
    """

    # save file
    file_name = "model_preds/bayesian_optim.txt"
    if os.path.isfile(file_name) and not refresh_log:
        with open(file_name, "a") as f:
            f.write(store_result)
    else:
        with open(file_name, "w") as f:
            f.write(store_result)


def read_train_data(TRAINING_DATA, STAGE, target_col):
    # data set
    if STAGE == 1:
        train = pd.read_csv(TRAINING_DATA)
    else:
        print("STAGE 2 Data")
        m_keys = dispatcher.MODELS.keys()
        len_key = len(m_keys)
        preds = pd.DataFrame()

        for i, k in enumerate(m_keys):
            pred = pd.read_csv(f"model_preds/{k}_pred.csv")
            print(i, k)
            columns = ["ID", "kfold"]
            if i == 0:
                preds = pred
            else:
                preds = pd.merge(preds, pred, on=columns, how="left")

        train = pd.read_csv(TRAINING_DATA)
        train = train.drop(DROP_COLS, axis=1)
        train = pd.merge(train, preds, on=columns, how="left")
        print(train.head())

    return train


if __name__ == "__main__":
    # read data set
    TARGET_COL = "Response"
    DROP_COLS = ["ID", "kfold", "Response"]
    df = read_train_data(TRAINING_DATA, STAGE=1, target_col=TARGET_COL)
    X = df.drop(DROP_COLS, axis=1)
    y = df[TARGET_COL]

    # BayesSearch
    print(f"Bayesian Optimization of {MODEL} model")
    result, trails = BayesSearch(X, y)

    # track best optimum result
    track_result(trails, MODEL, refresh_log)
