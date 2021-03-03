#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 09:13:14 2021


@author: sudhir
"""

# =============================================================================
# Import libary
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


# =============================================================================
# Dirichlet Encoder
# =============================================================================


class BetaEncoder(TransformerMixin, BaseEstimator):
    """
    Beta Encoder for Binary categorical variable

    moments = compute posterior mean and variance : ['mean','var']

    Reference:
    1. [Sampling Techniques in Bayesian Target Encoding  by Michael Larionov](
        https://arxiv.org/abs/2006.01317)
    2. [Encoding Categorical Variables with Conjugate Bayesian Models for WeWork
    Lead Scoring Engine by  Austin Slakey, Daniel Salas, and Yoni Schamroth](
        https://arxiv.org/abs/1904.13001)

    """

    def __init__(
        self,
        cat_cols=None,
        moments="mean",
        n_samples=10,
        sample_size=0.7,
        random_state=42,
        **kwarg,
    ):
        self.cat_cols = cat_cols
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.random_state = random_state
        self.moments = moments

    def fit(self, X, y):
        """fit"""
        if len(X) != len(y):
            print("recieved ", len(X), len(y))
            raise AssertionError("Length of X and y must be equal")

        # convert y to series:
        if type(y) == pd.DataFrame:
            y = y.iloc[:, 0]

        # fit alpha prior
        self._alpha_prior = np.mean(y)
        self._beta_prior = 1 - self._alpha_prior

        # _beta_pmf
        self._beta_pmf = dict()

        X_temp = X.copy(deep=True)
        target_col = "_y"
        X_temp[target_col] = y
        for feat_name, cat_col in self.cat_cols.items():
            # print(feat_name, cat_col)
            self._params_cat_col(cat_col, target_col, feat_name, X_temp)

        return self

    def _params_cat_col(self, cat_col, target_col, feat_name, X_temp):

        if isinstance(cat_col, list):
            columns = cat_col + [target_col]
            # type conversion
            for c in cat_col:
                X_temp[c] = X_temp[c].astype("object")
        else:
            columns = [cat_col, target_col]
            # type conversion
            X_temp[cat_col] = X_temp[cat_col].astype("object")

        ALL_LEVELS = X_temp[columns].groupby(cat_col).count().reset_index()

        for i in range(self.n_samples):
            X_sample = X_temp.sample(
                n=int(len(X_temp) * self.sample_size),
                replace=True,
                random_state=self.random_state + i,
            )

            # get positive count, full count
            beta = (
                X_sample[columns]
                .groupby(cat_col)
                .agg(fcount=(target_col, "count"), pcount=(target_col, "sum"))
                .reset_index()
            )

            # add prior to posterior
            beta["_alpha"] = self._alpha_prior + beta["pcount"]
            beta["_beta"] = self._beta_prior + beta["fcount"] - beta["pcount"]

            # fill NAs with prior
            beta = pd.merge(ALL_LEVELS, beta, on=cat_col, how="left")
            beta["_alpha"] = beta["_alpha"].fillna(self._alpha_prior)
            beta["_beta"] = beta["_beta"].fillna(self._beta_prior)

            if cat_col not in self._beta_pmf.keys():
                self._beta_pmf[cat_col] = beta[[cat_col, "_alpha", "_beta"]]
            else:
                self._beta_pmf[cat_col][["_alpha", "_beta"]] += beta[
                    ["_alpha", "_beta"]
                ]

        # report mean alpha and beta:
        self._beta_pmf[cat_col]["_alpha"] = (
            self._beta_pmf[cat_col]["_alpha"] / self.n_samples
        )
        self._beta_pmf[cat_col]["_beta"] = (
            self._beta_pmf[cat_col]["_beta"] / self.n_samples
        )

    def transform(self, X):

        X_temp = X.copy(deep=True)

        for feat_name, cat_col in self.cat_cols.items():

            if feat_name not in self._beta_pmf.keys():
                raise AssertionError(f"Column {cat_col} not fit by Beta ecoder")

            if isinstance(cat_col, list):
                # type conversion
                for c in cat_col:
                    X_temp[c] = X_temp[c].astype("object")
            else:
                # type conversion
                X_temp[cat_col] = X_temp[cat_col].astype("object")

            # add `_alpha` and `_beta` columns vi lookups, impute with prior
            X_temp = X_temp.merge(
                self._beta_pmf[cat_col],
                on=[cat_col],
                how="left",
            )

            X_temp["_alpha"] = X_temp["_alpha"].fillna(self._alpha_prior)
            X_temp["_beta"] = X_temp["_beta"].fillna(self._beta_prior)

            #   encode with moments
            if "mean" in self.moments:
                X_temp[cat_col + "__M"] = X_temp["_alpha"] / (
                    X_temp["_alpha"] + X_temp["_beta"]
                )
            if "variance" in self.moments:
                X_temp[cat_col + "__V"] = (X_temp["_alpha"] * X_temp["_beta"]) / (
                    ((X_temp["_alpha"] + X_temp["_beta"]) ** 2)
                    * (X_temp["_alpha"] + X_temp["_beta"] + 1)
                )
            # and drop columns
            # X_temp = X_temp.drop([cat_col], axis=1)
            X_temp = X_temp.drop(["_alpha"], axis=1)
            X_temp = X_temp.drop(["_beta"], axis=1)
        return X_temp
