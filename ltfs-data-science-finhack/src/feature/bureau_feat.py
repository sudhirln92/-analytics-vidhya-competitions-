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
import re
from sklearn.base import TransformerMixin, BaseEstimator

# =============================================================================
# Bureau feature
# =============================================================================


class BureauFeat(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        # super().__init__()
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        Xo = X.copy()

        # date_time_feature
        Xo = self.date_time_feature(Xo)

        Xfeat = pd.DataFrame(Xo["ID"].unique(), columns=["ID"])
        # SELF-INDICATOR

        Xg = (
            Xo.groupby(["ID"])
            .agg(cnt=("SELF-INDICATOR", "sum"))
            .rename(columns={"cnt": "SELF-INDICATOR"})
        )
        # left join
        Xfeat = pd.merge(Xfeat, Xg, how="left", on="ID")

        # hist amount
        famount = lambda x: [0 if len(w) == 0 else float(w) for w in x.split(",")]
        fsum = lambda x: sum(x)
        hist_cols = {
            "AMT PAID - HIST": "AMT_PAID_HIST",
            "AMT OVERDUE - HIST": "AMT_OVERDUE_HIST",
            "CUR BAL - HIST": "CUR_BAL_HIST",
        }
        Xo = Xo.rename(columns=hist_cols)

        for key, col in hist_cols.items():
            Xo[col] = Xo[col].fillna(",,,")
            Xo[col] = Xo[col].apply(famount)
            # compute stats
            Xo[col + "_sum"] = Xo[col].apply(fsum)

        # drop hist_cols

        # categorical columns
        cat_cols = [
            "MATCH-TYPE",
            "ACCT-TYPE",
            "CONTRIBUTOR-TYPE",
            "ACCOUNT-STATUS",
            "OWNERSHIP-IND",
            "ASSET_CLASS",
            "INSTALLMENT-FREQUENCY",
        ]

        for col in cat_cols:
            Xg = self.group_summary(Xo, col, dtype="cat")
            # merge
            Xfeat = pd.merge(Xfeat, Xg, how="left", on="ID")

        # fill na
        Xfeat = Xfeat.fillna(0)

        # numeric columns
        num_col = [
            "INSTALLMENT-AMT",
            "CREDIT-LIMIT/SANC AMT",
            "DISBURSED-AMT/HIGH CREDIT",
            "CURRENT-BAL",
            "OVERDUE-AMT",
            "WRITE-OFF-AMT",
            "TENURE",
            "AMT_PAID_HIST_sum",
            "AMT_OVERDUE_HIST_sum",
            "CUR_BAL_HIST_sum",
            # "month_DISBURSED-DT",
            # "month_CLOSE-DT",
            # "month_LAST-PAYMENT-DATE",
        ]

        for col in num_col:
            Xg = self.group_summary(Xo, col, dtype="num")
            # merge
            Xfeat = pd.merge(Xfeat, Xg, how="left", on="ID")

        print("missing value", Xfeat.isnull().sum().sum())
        return Xfeat

    def group_summary(self, X, col, dtype="cat"):
        "dtype : cat | num"

        # clean category
        f1 = lambda w: w if pd.isnull(w) else re.sub("(\W+)|( +)", "_", w)
        f2 = lambda x: x if pd.isnull(x) else re.sub("[^0-9]", "", x)

        if dtype == "cat":
            X[col] = X[col].apply(f1)
            # X[col] = X[col].fillna('MissingValue')
            Xg = (
                X.groupby(["ID", col])
                .agg(cnt=(col, "count"))
                .unstack(1)
                .droplevel(0, axis=1)
                .fillna(0)
            )
            col = f1(col)
            Xg.columns = [col + "__" + w for w in Xg.columns.values]

        else:
            # clean data
            if X[col].dtype == np.object:
                X[col] = X[col].apply(f2).fillna(0).astype("float32")
            # numeric agg
            Xg = X.groupby(["ID"]).agg(mean=(col, "mean"), std=(col, "std")).fillna(0)
            # rename columns
            col = f1(col)
            Xg.columns = [col + "__" + w for w in Xg.columns.values]

        return Xg

    def date_time_feature(self, X):

        Xo = X.copy()

        # days computation func
        col = ""
        col1 = ""

        def month_calculate(x):
            if x[[col, col1]].isnull().sum() > 0:
                val = 0
            else:
                try:
                    val = (x[col] - x[col1]) / pd.Timedelta(days=30)
                except:
                    val = 0
            return val

        # columns
        date_cols = ["DATE-REPORTED", "DISBURSED-DT", "CLOSE-DT", "LAST-PAYMENT-DATE"]
        col = "DATE-REPORTED"
        # Xo[col] = pd.to_datetime(Xo[col])
        # for col1 in date_cols[1:]:
        #     Xo[col1] = pd.to_datetime(Xo[col1])
        #     Xo["month_" + col1] = Xo.apply(month_calculate, axis=1)

        # hist datetime
        col = "DATE-REPORTED"
        col1 = "REPORTED DATE - HIST"

        def multimonth_calculate(x):
            if x[[col, col1]].isnull().sum() > 0:
                val = 0
            else:
                try:
                    val = [
                        (x[col] - pd.to_datetime(w)) / pd.Timedelta(days=30)
                        for w in x[col1].split(",")
                        if len(w) == 8
                    ]
                except:
                    val = 0
            return val

        # Xo["date_REPORTED_DATE_HIST"] = Xo.apply(multimonth_calculate, axis=1)
        fmean = lambda x: 0 if x == 0 else np.mean(x)
        # Xo["date_REPORTED_DATE_HIST_mean"] = Xo["date_REPORTED_DATE_HIST"].apply(fmean)

        # drop columns
        Xo = Xo.drop(date_cols, axis=1)
        Xo = Xo.drop(["REPORTED DATE - HIST"], axis=1)
        return Xo