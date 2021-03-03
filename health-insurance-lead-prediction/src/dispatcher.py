#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 09:13:14 2021


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
        objective="binary",
        boosting_type="gbdt",
        random_state=seed,
        learning_rate=0.01,
        n_estimators=1000,
        n_jobs=-1,
        **{
            "colsample_bytree": 0.4239880979970021,
            "max_depth": 10,
            "min_split_gain": 0.3104957653393776,
            "num_leaves": 34,
            "reg_alpha": 0.5698676660355981,
            "reg_lambda": 1.3119535973464989,
            "subsample": 0.8413327156973907,
        }
    ),
}


tmpe2 = {
    "lgbm": lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        random_state=seed,
        n_jobs=-1,
        learning_rate=0.07597559350356489,
        colsample_bytree=0.7457037476280491,
        subsample=0.7775574793447726,
        n_estimators=759,
        max_depth=6,
        num_leaves=1000,
        reg_alpha=0.27144710219397306,
        reg_lambda=0.8741205416771373,
    ),
    "xgbm": xgb.XGBClassifier(
        objective="binary:logistic",
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
