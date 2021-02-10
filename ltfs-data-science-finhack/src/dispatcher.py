#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 09:13:14 2021


@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
import lightgbm as lgb
import xgboost as xgb
from .utils.file_handler import read_config

config = read_config("config.json")
seed = config["seed"]
# =============================================================================
# MODELS
# =============================================================================

MODELS = {
    "lgbm": lgb.LGBMClassifier(
        **{
            "objective": "multiclass",
            "boosting_type": "gbdt",
            "num_class": 7,
            "random_state": seed,
            "verbose": 0,
            "nthread": -1,
            "colsample_bytree": 0.9723552887799483,
            "learning_rate": 0.07419873126121708,
            "max_depth": 8,
            "n_estimators": 298,
            "num_leaves": 7700,
            "reg_alpha": 0.21928108502014154,
            "reg_lambda": 0.8046328503330902,
            "subsample": 0.8983872887835846,
        }
    )
}

{
    "objective": "multiclass",
    "boosting_type": "gbdt",
    "num_class": 7,
    "random_state": seed,
    "verbose": 0,
    "nthread": -1,
    "colsample_bytree": 0.8,
    "learning_rate": 0.0484,
    "max_depth": 10,
    "min_child_samples": 47,
    "min_split_gain": 0.883,
    "n_estimators": 1000,
    "num_leaves": 2900,
    "reg_alpha": 0.45107,
    "reg_lambda": 0.3808,
    "subsample": 0.7072,
}


MODELS_STAGE2 = {}

tmpe2 = {
    "lgbm": {
        "objective": "multiclass",
        "boosting_type": "gbdt",
        "num_class": 7,
        "random_state": seed,
        "colsample_bytree": 0.7773032534884761,
        "learning_rate": 0.09282250301860116,
        "max_depth": 12,
        "min_child_samples": 82,
        "min_split_gain": 0.2988930976539875,
        "n_estimators": 325,
        "num_leaves": 9000,
        "reg_alpha": 0.5600803851920294,
        "reg_lambda": 0.11094228430582374,
        "subsample": 0.8834598837438516,
        "is_unbalance": True,
        "verbose": 100,
        "nthread": -1,
    },
}


tmp = {
    "xgbm": xgb.XGBClassifier(
        objective="multi:softmax",
        learning_rate=0.08013538134858043,
        colsample_bytree=0.9338376427758461,
        subsample=0.8167099091109269,
        max_depth=8,
        gamma=4,
        n_estimators=937,
        reg_alpha=0.020365987648895845,
        reg_lambda=0.22136606795195418,
        random_state=seed,
        nthread=-1,
        eval_metric="mlogloss",
    ),
    "rftree": ensemble.RandomForestClassifier(
        criterion="entropy",
        bootstrap=True,
        n_estimators=867,
        min_samples_leaf=38,
        min_samples_split=28,
        max_depth=14,
        max_features="auto",
        n_jobs=-1,
        random_state=seed,
    ),
}
