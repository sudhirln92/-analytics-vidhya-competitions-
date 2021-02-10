#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 09:13:14 2021


@author: sudhir
"""

# =============================================================================
# Import libary
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import KFold
from sklearn import preprocessing


# =============================================================================
# Dirichlet Encoder
# =============================================================================


class DirichletEncoder(TransformerMixin, BaseEstimator):
    """
    Dirichlet Encoder for multinomial categorical variable

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
        self.alpha_prior = dict()
        if len(self.alpha_prior.keys()) == 0:
            temp = y.value_counts().to_dict()
            full_count = len(y)
            self.alpha_prior = {k: temp[k] / full_count for k in temp}

        # _dirichlet_distributions
        self._dirichlet_distributions = dict()

        X_temp = X.copy(deep=True)
        target_col = "_y"
        X_temp[target_col] = y
        for feature_name, cat_col in self.cat_cols.items():
            # print(feature_name, cat_col)
            self._alphas_cat_col(cat_col, target_col, feature_name, X_temp)

        return self

    def _alphas_cat_col(self, cat_col, target_col, feature_name, X_temp):

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

            # alphas
            alpha_dicts = dict()

            for k in self.alpha_prior.keys():

                # prior for dirichlet didtribution
                prior = self.alpha_prior[k]
                feature_name_k = feature_name + "_N" + str(k)

                # get positive sample
                alpha_k = (
                    X_sample[columns]
                    .query(f"{target_col} == {k}")
                    .groupby(cat_col)
                    .agg(N=(target_col, "count"))
                    .rename(columns={"N": feature_name_k})
                    .reset_index()
                )

                # add prior
                alpha_k[feature_name_k] += prior

                # merge to all unique values
                alpha_k = pd.merge(ALL_LEVELS, alpha_k, on=cat_col, how="left")
                alpha_k = alpha_k.fillna(prior)

                # save
                if isinstance(cat_col, list):
                    columns_alpha = cat_col + [feature_name_k]
                else:
                    columns_alpha = [cat_col, feature_name_k]
                alpha_dicts[k] = alpha_k[columns_alpha]

            # dirichlet distribution coef
            if feature_name not in self._dirichlet_distributions.keys():
                self._dirichlet_distributions[feature_name] = alpha_dicts
            else:
                for k in alpha_dicts.keys():
                    feature_name_k = feature_name + "_N" + str(k)
                    self._dirichlet_distributions[feature_name][k][
                        feature_name_k
                    ] += alpha_dicts[k][feature_name_k]
        # for last loop
        for k in alpha_dicts.keys():
            feature_name_k = feature_name + "_N" + str(k)
            self._dirichlet_distributions[feature_name][k][feature_name_k] = (
                self._dirichlet_distributions[feature_name][k][feature_name_k]
                / self.n_samples
            )

    def transform(self, X):

        X_temp = X.copy(deep=True)

        for feature_name, cat_col in self.cat_cols.items():

            if feature_name not in self._dirichlet_distributions.keys():
                raise AssertionError(f"Column {cat_col} not fit by Dirichlet encoder")

            if isinstance(cat_col, list):
                # type conversion
                for c in cat_col:
                    X_temp[c] = X_temp[c].astype("object")
            else:
                # type conversion
                X_temp[cat_col] = X_temp[cat_col].astype("object")

            # a0= sum of all alpha
            X_temp[feature_name + "_a0"] = 0

            # add alpha_k via
            alphas = self._dirichlet_distributions[feature_name]
            for k in alphas.keys():
                X_temp = X_temp.merge(alphas[k], on=cat_col, how="left")
                feature_name_alpha = feature_name + "_alpha_" + str(k)
                feature_name_k = feature_name + "_N" + str(k)
                X_temp[feature_name_alpha] = X_temp[feature_name_k].fillna(
                    self.alpha_prior[k]
                )
                X_temp[feature_name + "_a0"] += X_temp[
                    feature_name + "_alpha_" + str(k)
                ]
            # compute posterior mean and variance
            for k in alphas.keys():

                if "mean" in self.moments:
                    feature_name_k = feature_name + "_N" + str(k)
                    X_temp[feature_name + "_M_" + str(k)] = (
                        X_temp[feature_name + "_alpha_" + str(k)]
                        / X_temp[feature_name + "_a0"]
                    )
                if "var" in self.moments:
                    feature_name_k = feature_name + "_N" + str(k)
                    X_temp[feature_name + "_V_" + str(k)] = (
                        X_temp[feature_name + "_alpha_" + str(k)]
                        / X_temp[feature_name + "_a0"]
                    )
                # drop alpha_k and count
                feature_name_k = feature_name + "_N" + str(k)
                X_temp = X_temp.drop(feature_name_k, axis=1)
                feature_name_alpha = feature_name + "_alpha_" + str(k)
                X_temp = X_temp.drop(feature_name_alpha, axis=1)

            # now drop category columns
            # X_temp = X_temp.drop(cat_col, axis=1)
            X_temp = X_temp.drop(feature_name + "_a0", axis=1)
            # drop one kth encoded feature
            feature_name_k = feature_name + "_M_" + str(max(self.alpha_prior.keys()))
            X_temp = X_temp.drop(feature_name_k, axis=1)

        return X_temp
