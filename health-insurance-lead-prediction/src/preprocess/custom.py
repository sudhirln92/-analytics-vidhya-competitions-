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
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Drop Column
# =============================================================================


class DropColumn(TransformerMixin, BaseEstimator):
    """Drop Columns from original Data Frame:

    Parameters
    ----------
    columns: list, The list of column names.

    Returns
    -------
    DataFrame

    Size: (row, old columns - columns)
    """

    def __init__(self, cols):
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xdrop = X.drop(self.cols, axis=1)
        return Xdrop


# =============================================================================
# Extract Column
# =============================================================================


class ExtractColumn(TransformerMixin, BaseEstimator):
    """Extract Columns from original Data Frame:

    Parameters
    ----------
    columns: list, The list of column names.

    Returns
    -------
    DataFrame with only extracted columns

    Size: (row, columns)
    """

    def __init__(self, columns, **kwargs):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xcol = X[self.columns]
        return Xcol
