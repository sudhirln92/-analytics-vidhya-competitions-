#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 09:13:14 2021


@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import impute
import joblib

seed = 42
# =============================================================================
# Custom Imputer
# =============================================================================


class MissForestImputer(TransformerMixin, BaseEstimator):
    """Miss Forest
    We fill missing value using miss forest algorithm. Initially, we identify
    the number of missing value in all the columns and sort by the count.
    That column with the least missing number of values is filled first. We
    split data into observed and missed data set. The target column without
    missing value is utilized to train the random forest algorithm. The later
    trained model is used to fill the missing value in the missed data-target
    column. Initial imputation is using with the mean for continues variable
    and with mode for the categorical variable.

    Reference: [MissForest - nonparametric missing value imputation for
    mixed-type data. by Daniel J. Stekhoven and Peter Bühlmann](
        https://arxiv.org/abs/1105.0828)
    """

    def __init__(
        self,
        cat_cols,
        num_cols,
        date_cols,
        dont_use_cols,
        n_estimators,
        max_depth,
        seed=42,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.date_cols = date_cols
        self.dont_use_cols = dont_use_cols
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y=None):
        # Miss Forest
        Xo = X.copy()
        self._find_mean_mode(Xo)

        # call miss forest
        self.Xfill = self._miss_forest(Xo)
        return self

    def transform(self, X):
        # fill missing value in date column
        # X = self.fill_missing_value_datetime(X)

        # pickle model file and save memory
        self.model = joblib.load(f"models/custom_impute_model.pkl")

        # call miss forest
        Xo = self._miss_forest_predict(X)

        del self.model

        return Xo

    def fill_missing_value_datetime(self, X):
        """Filll missing value in date time columns"""
        if isinstance(self.date_cols, dict):
            for c in self.date_cols:
                X[c] = pd.to_datetime(X[c])
                X[c] = X[c].fillna(self.date_cols[c])
        else:
            for c in self.date_cols:
                X[c] = pd.to_datetime(X[c])
                X[c] = X[c].fillna(pd.Timestamp("today"))
        return X

    def _find_mean_mode(self, X):
        """
        Find mean for numeric variable
        Find mode for categorical variable
        """
        self.col_modes = {}
        self.col_means = {}
        # mode
        for c in self.cat_cols:
            self.col_modes[c] = X[c].mode()[0]
        # mean
        for c in self.num_cols:
            self.col_means[c] = X[c].mean()

    def _miss_forest(self, X):

        self.model = {}
        self.encoder = {}
        # prepare
        if isinstance(self.date_cols, dict):
            drop_col = self.dont_use_cols + list(self.date_cols.keys())
        else:
            drop_col = self.dont_use_cols + self.date_cols
        Xcp = X.copy()
        Xo = Xcp.drop(drop_col, axis=1)

        cnt_missing = Xo.isnull().sum().sort_values()
        missing_col = list(cnt_missing[cnt_missing > 0].index)

        for col in missing_col:
            Ximp = Xo.drop(col, axis=1)
            yimp = Xo[col]

            missing_idx = Xo[Xo[col].isnull()].index
            present_idx = Xo[Xo[col].notnull()].index

            # fill mean and mode
            dummy_col = [w for w in self.cat_cols if w != col]
            for c in dummy_col:
                if c in self.cat_cols:
                    Ximp[c] = Ximp[c].fillna(self.col_modes[c])
            num_col_oops = [w for w in self.num_cols if w != col]
            for c in num_col_oops:
                Ximp[c] = Ximp[c].fillna(self.col_means[c])

            Ximp = pd.get_dummies(Ximp, columns=dummy_col, drop_first=True)

            # train test split
            Xtr = Ximp.loc[present_idx].reset_index(drop=True)
            ytr = yimp.loc[present_idx].reset_index(drop=True)
            test = Ximp.loc[missing_idx].reset_index(drop=True)
            # train model
            if col in self.cat_cols:
                # label encoder
                label_encoder = preprocessing.LabelEncoder().fit(ytr)
                self.encoder[col] = label_encoder
                ytr = label_encoder.transform(ytr)
                rf_classifier = ensemble.RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    criterion="gini",
                    max_depth=self.max_depth,
                    n_jobs=-1,
                    oob_score=True,
                    random_state=self.seed,
                ).fit(Xtr, ytr)
                self.model[col] = rf_classifier

                # predict
                pred = rf_classifier.predict(test)
                # label encoder
                pred = label_encoder.inverse_transform(pred)
                Xo.loc[missing_idx, col] = pred
                print(f"oob_score {rf_classifier.oob_score_} for columns {col}")
            else:
                rf_regressor = ensemble.RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    criterion="mse",
                    max_depth=self.max_depth,
                    n_jobs=-1,
                    oob_score=True,
                    random_state=self.seed,
                ).fit(Xtr, ytr)

                self.model[col] = rf_regressor
                pred = rf_regressor.predict(test)
                Xo.loc[missing_idx, col] = pred
                print(f"oob_score {rf_regressor.oob_score_} for columns {col}")

        # pickle model file and save memory
        joblib.dump(self.model, f"models/custom_impute_model.pkl")

        del self.model

        # concate dont use col
        Xo = pd.concat([Xcp[drop_col], Xo], axis=1)
        return Xo

    def _miss_forest_predict(self, X):

        # prepare
        if isinstance(self.date_cols, dict):
            drop_col = self.dont_use_cols + list(self.date_cols.keys())
        else:
            drop_col = self.dont_use_cols + self.date_cols
        Xcp = X.copy()
        Xo = Xcp.drop(drop_col, axis=1)

        cnt_missing = Xo.isnull().sum()  # .sort_values()
        missing_col = list(cnt_missing[cnt_missing > 0].index)

        # fill missing value
        for col in missing_col:
            Xts = Xo.drop(col, axis=1)
            yts = Xo[col]

            missing_idx = Xo[Xo[col].isnull()].index
            present_idx = Xo[Xo[col].notnull()].index

            # fill mean and mode
            dummy_col = [w for w in self.cat_cols if w != col]
            for c in dummy_col:
                if c in self.cat_cols:
                    Xts[c] = Xts[c].fillna(self.col_modes[c])
            num_col_oops = [w for w in self.num_cols if w != col]
            for c in num_col_oops:
                Xts[c] = Xts[c].fillna(self.col_means[c])

            # create dummy columns
            Xts = pd.get_dummies(Xts, columns=dummy_col, drop_first=True)

            # test data
            Xts = Xts.loc[missing_idx].reset_index(drop=True)
            if col in self.cat_cols:
                if col in self.model:
                    rf_classifier = self.model[col]
                    pred = rf_classifier.predict(Xts)
                    pred = self.encoder[col].inverse_transform(pred)
                    Xo.loc[missing_idx, col] = pred
                else:
                    Xo.loc[missing_idx, col] = self.col_modes[col]
            else:
                if col in self.model:
                    rf_regressor = self.model[col]
                    pred = rf_regressor.predict(Xts)
                    Xo.loc[missing_idx, col] = pred
                else:
                    Xo.loc[missing_idx, col] = self.col_means[col]

        # concate dont use col
        Xo = pd.concat([Xcp[drop_col], Xo], axis=1)
        return Xo

    def fit_transform(self, X, y=None):
        " miss forest fit trasform"
        if y is not None:
            # fill missing value in date column
            X = self.fill_missing_value_datetime(X)

            # for train data
            self.fit(X)

            # no need of trasform
            # print(self.Xfill.isnull().sum())
            return self.Xfill
        else:
            return self.fit(X, y).transform(X)


# =============================================================================
# Custom Imputer
# =============================================================================


class CustomKNNImputer(TransformerMixin, BaseEstimator):
    """KNN imputer"""

    def __init__(self, cat_cols=None, num_cols=None, n_neighbors=5, weights="distance"):
        if isinstance(num_cols, str):
            self.num_cols = [num_cols]
        else:
            self.num_cols = num_cols
        if isinstance(cat_cols, str):
            self.cat_cols = cat_cols
        else:
            self.cat_cols = cat_cols
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y=None):
        if self.cat_cols is not None:
            Xo = X[self.cat_cols].copy()
            self.knn_cat = impute.KNNImputer(
                n_neighbors=self.n_neighbors, weights=self.weights
            )
            self.knn_cat.fit(Xo)
        if self.num_cols is not None:
            Xo = X[self.num_cols].copy()
            self.knn_num = impute.KNNImputer(
                n_neighbors=self.n_neighbors, weights=self.weights
            )
            self.knn_num.fit(Xo)

        return self

    def transform(self, X):
        if self.cat_cols is not None:
            Xo = X[self.cat_cols].copy()
            Xo = self.knn_cat.transform(Xo)
            X[self.cat_cols] = pd.DataFrame(Xo)
        if self.num_cols is not None:
            Xo = X[self.num_cols].copy()
            Xo = self.knn_num.transform(Xo)
            X[self.num_cols] = pd.DataFrame(Xo)
        return X
