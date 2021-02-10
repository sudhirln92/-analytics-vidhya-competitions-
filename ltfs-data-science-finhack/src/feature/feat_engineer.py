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


class DateTransform(TransformerMixin, BaseEstimator):
    """Datetime Feature extraction from Reporting Date feature:

    Parameters
    ----------
    column: string, The column name.

    Returns
    -------
    DataFrame with column names based on new datatime feature extracted

    Size: (row, new_columns)
    """

    def __init__(self, reference_date=None, date_cols=None, **kwargs):
        self.date_cols = date_cols
        self.reference_date = reference_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """DatetimeFeature
        'AuthDate','DisbursalDate','MaturityDAte'
        """
        Xo = X.copy()

        # fill na
        date_col = ["AuthDate", "DisbursalDate", "MaturityDAte"]

        for col in date_col:
            Xo[col] = Xo[col].fillna(self.reference_date)
            Xo[col] = pd.to_datetime(Xo[col])
        # date feature
        Xo = self.datetime_feature(Xo, "AuthDate")
        Xo = self.datetime_feature(Xo, "MaturityDAte")

        Xo["disbursa_months"] = (Xo["AuthDate"] - Xo["DisbursalDate"]) / pd.Timedelta(
            days=30
        )

        # Drop col
        # date_col = ["AuthDate", "DisbursalDate", "MaturityDAte"]
        # for col in date_col:
        #     Xo[col] = Xo[col].dt.strftime("%Y%m%d").astype("int64")
        # X = X.drop(cols, axis=1)
        # Xo = pd.concat([X, Xo], axis=1)
        return Xo

    def datetime_feature(self, X, col):
        # X[col + "_date"] = X[col].dt.strftime("%Y%m%d").astype("int64")
        # X[col + "_time"] = X[col].dt.hour * 60 + X[col].dt.minute
        # X[col + "_hour"] = X[col].dt.hour
        # X[col+'_minute'] = X[col].dt.minute
        X[col + "_day"] = X[col].dt.day.astype("uint16")
        X[col + "_weekday"] = X[col].dt.weekday.astype("uint16")  # Start from monday
        X[col + "_month"] = X[col].dt.month.astype("uint16")
        X[col + "_year"] = X[col].dt.year.astype("uint16")
        return X
