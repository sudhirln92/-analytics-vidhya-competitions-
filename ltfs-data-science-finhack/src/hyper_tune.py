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
from .skmetrics import evalerror
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
    elif MODEL == "sgd":
        model = linear_model.SGDClassifier(**params, random_state=seed)
    elif MODEL == "rftree":
        model = ensemble.RandomForestClassifier(**params, n_jobs=-1, random_state=seed)
    elif MODEL == "extree":
        model = ensemble.ExtraTreesClassifier(**params, n_jobs=-1, random_state=seed)
    elif MODEL == "gbm":
        model = ensemble.GradientBoostingClassifier(**params, random_state=seed)
    elif MODEL == "knn":
        model = neighbors.KNeighborsClassifier(**params, n_jobs=-1)
    elif MODEL == "lgbm":
        model = lgb.LGBMClassifier(
            **params,
            objective="multiclass",
            boosting_type="gbdt",
            num_class=3,
            random_state=seed,
        )
    elif MODEL == "xgbm":
        model = xgb.XGBClassifier(
            **params,
            objective="multi:softmax",
            nthread=-1,
            random_state=seed,
            eval_metric="mlogloss",
        )

    # initialise stratified k-fold
    kf = model_selection.StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)
    accuracies = []
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
                eval_metric=evalerror,
                verbose=25,
                early_stopping_rounds=20,
            )
        else:
            model.fit(X_train, y_train)
        # predict
        y_prob = model.predict_proba(X_valid)
        y_pred = y_prob.argmax(axis=1)
        fold_acc = metrics.log_loss(y_valid, y_prob)
        fold_acc = metrics.f1_score(y_valid, y_pred, average="macro")
        print("f1_score", fold_acc)
        accuracies.append(fold_acc)

    print("accuracies:\t", accuracies)
    return -np.mean(accuracies)


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
    elif MODEL == "sgd":
        param_space = {
            "loss": hp.choice("loss", ["log", "modified_huber"]),
            "penalty": hp.choice("penalty", ["l2", "l1", "elasticnet"]),
            "alpha": scope.float(hp.uniform("alpha", 0.001, 1)),
            "max_iter": scope.int(hp.uniform("max_iter", 100, 1500)),
        }
    elif MODEL == "rftree":
        param_space = {
            "max_depth": scope.int(hp.quniform("max_depth", 6, 15, 1)),
            "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1000, 1)),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "max_features": hp.choice("max_features", ["auto", "log2"]),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 6, 100, 1)),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 6, 100, 1)),
            #'bootstrap': hp.choice('bootstrap', [True, False]),
        }
    elif MODEL == "extree":
        param_space = {
            "max_depth": scope.int(hp.quniform("max_depth", 5, 25, 1)),
            "n_estimators": scope.int(hp.quniform("n_estimators", 100, 2000, 1)),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "max_features": hp.choice("max_features", ["auto", "log2"]),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 3, 100, 1)),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 3, 100, 1)),
            #'bootstrap': hp.choice('bootstrap', [True, False]),
        }
    elif MODEL == "gbm":
        param_space = {
            "learning_rate": scope.float(hp.uniform("learning_rate", 0.001, 1)),
            "n_estimators": scope.int(hp.quniform("n_estimators", 100, 2000, 1)),
            "subsample": scope.float(hp.uniform("subsample", 0.001, 1)),
            "criterion": hp.choice("criterion", ["friedman_mse", "mse", "mae"]),
            "max_features": hp.choice("max_features", ["auto", "log2"]),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 3, 100, 1)),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 3, 100, 1)),
            # 'loss':hp.choice('bootstrap',['deviance', 'exponential']),
        }
    elif MODEL == "knn":
        param_space = {
            "n_neighbors": scope.int(hp.quniform("n_neighbors", 5, 100, 1)),
            "leaf_size": scope.int(hp.quniform("leaf_size", 30, 200, 1)),
        }
    elif MODEL == "lgbm":
        param_space = {
            "learning_rate": scope.float(hp.uniform("learning_rate", 0.0001, 0.1)),
            "n_estimators": scope.int(hp.quniform("n_estimators", 25, 1000, 1)),
            "max_depth": scope.int(hp.quniform("max_depth", 6, 15, 1)),
            "subsample": scope.float(hp.uniform("subsample", 0.6, 1)),
            "colsample_bytree": scope.float(hp.uniform("colsample_bytree", 0.6, 1)),
            # "subsample_freq":scope.int(hp.quniform("subsample_freq", 0, 5, 1)),
            # "min_child_samples": scope.int(
            #     hp.quniform("min_child_samples", 20, 100, 1)
            # ),
            # "min_split_gain": scope.float(hp.uniform("min_split_gain", 0.01, 1)),
            "reg_alpha": scope.float(hp.uniform("reg_alpha", 0.0001, 1)),
            "reg_lambda": scope.float(hp.uniform("reg_lambda", 0.0001, 1)),
            "num_leaves": scope.int(hp.quniform("num_leaves", 32, 10000, 100)),
        }
    elif MODEL == "xgbm":
        param_space = {
            "learning_rate": scope.float(hp.uniform("learning_rate", 0.0001, 0.1)),
            "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1000, 1)),
            "max_depth": scope.int(hp.quniform("max_depth", 6, 10, 1)),
            "subsample": scope.float(hp.uniform("subsample", 0.7, 1)),
            "colsample_bytree": scope.float(hp.uniform("colsample_bytree", 0.7, 1)),
            "gamma": scope.int(hp.quniform("gamma", 0, 20, 1)),
            "reg_alpha": scope.float(hp.uniform("reg_alpha", 0.01, 1)),
            "reg_lambda": scope.float(hp.uniform("reg_lambda", 0.01, 1)),
            # "scale_pos_weight":scope.float(hp.uniform("scale_pos_weight", 0.001, 1)),
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
    TARGET_COL = "Top-up Month"
    DROP_COLS = ["ID", "kfold", "Top-up Month"]
    df = read_train_data(TRAINING_DATA, STAGE=1, target_col=TARGET_COL)
    X = df.drop(DROP_COLS, axis=1)
    y = df[TARGET_COL]

    # BayesSearch
    print(f"Bayesian Optimization of {MODEL} model")
    result, trails = BayesSearch(X, y)

    # track best optimum result
    track_result(trails, MODEL, refresh_log)
