#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 09:13:14 2021


@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
import re
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer


class Preprocess(BaseEstimator, TransformerMixin):
    def __init__(self):
        # super().__init__()
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # clean payment
        pay_clean = {
            "Cheque": "PDC",
            "Auto Debit": "Billed",
            "PDC_E": "PDC",
            "SI Reject": "Reject",
            "ECS Reject": "Reject",
            "PDC Reject": "Reject",
            "Escrow": "Direct Debit",
        }

        X["PaymentMode"] = X["PaymentMode"].replace(pay_clean)

        # log transform
        X["MonthlyIncome"] = X["MonthlyIncome"].apply("log1p")
        return X


# =============================================================================
# Handle Outlier
# =============================================================================


class HandleOutlier(TransformerMixin, BaseEstimator):
    """HandleOutlier in columns:

    Parameters
    ----------
    Input dataframe,
    columns names: ['PickUpLat','PickUpLat']
    new_outlier_col: True | False True add as new columns False then replace with missing value

    Returns
    -------
    DataFrame

    Size: (row, columns)
    """

    def __init__(self, cols, new_outlier_col=False, **kwargs):
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.new_outlier_col = new_outlier_col

    def fit(self, X, y=None):
        # compute
        self.lower_bound = {}
        self.upper_bound = {}
        for col in self.cols:
            low, up = self.compute_bound(X, col)
            self.lower_bound[col] = low
            self.upper_bound[col] = up

        return self

    def compute_bound(self, X, col):
        """Compute lower bound and upper bound for upper bound
        for outlier detection
        """
        q1 = X[col].quantile(0.25)
        q2 = X[col].quantile(0.75)
        iqr = q2 - q1
        low = q1 - 1.5 * iqr
        up = q2 + 1.5 * iqr
        return low, up

    def transform(self, X):
        # print('Limit', self.limit)
        for col in self.cols:
            low = self.lower_bound[col]
            up = self.upper_bound[col]
            if self.new_outlier_col:
                fout = lambda x: x < low or x > up
                X[col + "_outlier"] = X[col].apply(fout).astype("uint8")
            else:
                fout = lambda x: np.nan if x < low or x > up else x
                X[col] = X[col].apply(fout)
        return X


# =============================================================================
# CustomQuantileTransformer
# =============================================================================


class CustomQuantileTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        cols=None,
        n_quantiles=1000,
        output_distribution="normal",
        random_state=42,
        **kwargs,
    ):
        """
        cols: pass column names
        n_quantiles:
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        fit
        """
        self.quant_trans = QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution,
            random_state=self.random_state,
        )
        if isinstance(X, pd.DataFrame):
            self.quant_trans.fit(X[self.cols])
        elif isinstance(X, np.ndarray):
            self.quant_trans.fit(X)
        else:
            raise ValueError("input should be DataFrame or array")
        return self

    def transform(self, X):
        """
        transform
        """
        if isinstance(X, pd.DataFrame):
            Xo = self.quant_trans.transform(X[self.cols])
            Xo = pd.DataFrame(Xo, columns=self.cols)
            Xo = pd.concat([X.drop(self.cols, axis=1), Xo], axis=1)
        elif isinstance(X, np.ndarray):
            Xo = self.quant_trans.transform(X)
        else:
            raise ValueError("input should be DataFrame or array")
        return Xo

    def inverse_transform(self, X):
        """
        inverse_transform
        """
        if isinstance(X, pd.DataFrame):
            Xo = self.quant_trans.inverse_transform(X[self.cols])
            Xo = pd.DataFrame(Xo, columns=self.cols)
            Xo = pd.concat([X.drop(self.cols, axis=1), Xo], axis=1)
        elif isinstance(X, np.ndarray):
            Xo = self.quant_trans.inverse_transform(X)
        else:
            raise ValueError("input should be DataFrame or array")

        return Xo


class TypeConversion(TransformerMixin, BaseEstimator):
    """The data type converion class will convert different columns
    into desired data type.

    Parameters
    ----------
    Numeric columns names: list of column names

    Categorical column names: list of column names

    Categorical columns with unsigned integer

    Returns
    -------
    DataFrames

    Size: (row, columns)
    """

    def __init__(
        self, num_cols=None, cat_cols=None, uint_cols=None, date_cols=None, **kwargs
    ):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.uint_cols = uint_cols
        self.date_col = date_cols

    def fit(self, X, y=None):
        self.category = {}
        for col in self.cat_cols:
            self.category[col] = X[col].astype("category").cat.categories
        return self

    def transform(self, X):
        """Data Type Conversion"""
        # float type columns
        if self.num_cols is not None:
            X[self.num_cols] = X[self.num_cols].apply(lambda x: x.astype("float32"))
        # unsigned integer
        if self.uint_cols is not None:
            X[self.uint_cols] = X[self.uint_cols].apply(lambda x: x.astype("uint16"))
        # datetime columns
        if self.date_col is not None:
            for col in self.date_col:
                X[col] = pd.to_datetime(X[col])

        # categorical columns
        if self.cat_cols is not None:
            # X[self.cat_cols] = X[self.cat_cols].apply(lambda x: x.astype('category'))
            for col in self.cat_cols:
                X[col] = pd.Categorical(X[col], categories=self.category[col])
        return X