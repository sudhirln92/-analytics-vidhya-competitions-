#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 09:13:14 2021


@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import preprocessing

seed = 42
# =============================================================================
# Custom Imputer
# =============================================================================


class SimpleImpute(TransformerMixin, BaseEstimator):
    """
    Simple imputation stargey based on data analysis
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # mode
        self.mode = X["Health Indicator"].mode()[0]

        return self

    def transform(self, X):
        X["Holding_Policy_Duration"] = X["Holding_Policy_Duration"].replace({"14+": 15})

        # impute missing value
        X["Health Indicator"] = X["Health Indicator"].fillna(self.mode)

        # few of customer do not have any policy subscribed
        X["Holding_Policy_Duration"] = X["Holding_Policy_Duration"].fillna(0)
        X["Holding_Policy_Type"] = X["Holding_Policy_Type"].fillna("no_policy")

        return X