#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 09:13:14 2021


@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from itertools import combinations


# =============================================================================
# Dummy variable creator / One hot encoding
# =============================================================================


class DummyTransformer(TransformerMixin, BaseEstimator):
    """Dummy Varaible Transform / One Hot Encoding:

    Parameters
    ----------
    columns: list, input Required columns name to get dummay variables


    Returns
    -------
    DataFrame with column names

    Size: (row, new_columns + old columns)
    """

    def __init__(self, columns, **kwargs):
        self.columns = columns

    def fit(self, X, y=None):
        self.cat = {}
        for col in self.columns:
            self.cat[col] = X[col].astype("category").cat.categories
        return self

    def transform(self, X):

        Xtmp = X[self.columns].copy()

        # transform categorical
        for col in self.columns:
            Xtmp[col] = pd.Categorical(Xtmp[col], categories=self.cat[col])
        Xcat = pd.get_dummies(Xtmp, columns=self.columns, drop_first=True)

        # Merge dataset
        Xtmp = pd.concat([X, Xcat], axis=1)
        return Xtmp


class BinFeat(TransformerMixin, BaseEstimator):
    """
    divide data into separate and one encode on the feature
    """

    def __init__(self, bins, **kwargs):
        self.bins = bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        for col, bins in self.bins.items():
            # bins
            Xbin = pd.cut(X[col], bins=bins, labels=False)
            # categories
            categories = range(1, len(bins) - 2)
            Xbin = pd.Categorical(Xbin, categories=categories)
            Xbin = pd.get_dummies(Xbin, drop_first=True, prefix=col + "_")
            # concat
            X = pd.concat([X, Xbin], axis=1)
        return X


class RatioFeat(TransformerMixin, BaseEstimator):
    """
    Ratios
    """

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # ratio of age
        X["premium_to_age1"] = X["Reco_Policy_Premium"] / X["Upper_Age"]
        X["premium_to_age2"] = X["Reco_Policy_Premium"] / X["Lower_Age"]

        # duration
        X["duration"] = X["Upper_Age"] - X["Lower_Age"]

        return X


class CombineCatFeat(TransformerMixin, BaseEstimator):
    """
    Combine Categorical columns
    create combination of columns
    apply frequency encode
    """

    def __init__(self, columns, encoder_suffix=None, **kwargs):
        self.columns = columns
        self.encoder_suffix = encoder_suffix

    def frequency_encode(self, X, column):
        # frequency_encode
        encoder = X.groupby(column).size() / len(X)
        return encoder

    def fit(self, X, y=None):
        """ create combination of columns """
        Xo = X.copy()
        cat_feature = combinations(self.columns, 2)
        self.combine_label_encoder = {}
        self.freq_encoder = {}
        self.mode = {}
        for cols in cat_feature:
            column = f"{cols[0]}" + "_" + f"{cols[1]}"
            Xo[column] = Xo[cols[0]].astype(str) + "_" + Xo[cols[0]].astype(str)
            # find mode
            self.mode[column] = Xo[column].mode()[0]
            # label encoder
            le = LabelEncoder().fit(Xo[column])
            self.combine_label_encoder[column] = le

            # apply transform
            Xo[column] = le.transform(Xo[column])
            self.freq_encoder[column] = self.frequency_encode(Xo, column)

        return self

    def transform(self, X):
        """Combine cateforical feature and apply frequency encode"""

        cat_feature = combinations(self.columns, 2)
        for cols in cat_feature:
            column = f"{cols[0]}" + "_" + f"{cols[1]}"
            X[column] = X[cols[0]].astype(str) + "_" + X[cols[0]].astype(str)
            # fill na
            X[column] = X[column].fillna(self.mode[column])
            # label encoding
            encoder = self.combine_label_encoder[column]
            X[column] = encoder.transform(X[column])
            # frequency encoder
            if self.encoder_suffix:
                new_col = column + "_" + self.encoder_suffix
                X = pd.merge(X, self.freq_encoder[column], on=column, how="left")
                X = X.rename(columns={column: new_col})
            else:
                freq_encoder = self.freq_encoder[column]
                X[column] = X[column].apply(lambda x: freq_encoder[x])
        return X
