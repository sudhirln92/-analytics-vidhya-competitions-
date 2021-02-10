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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.mixture import GaussianMixture
from sklearn import ensemble
import joblib

from .preprocess import preprocess, custom
from .preprocess import custom_impute
from .feature import encoder, feat_engineer, bureau_feat, demography
from .utils.file_handler import read_config


config = read_config("config.json")
seed = config["seed"]
# =============================================================================
# pipe
# =============================================================================


class CustomPipeline:
    """
    Custom Pipeline
    """

    def __init__(self, pickle_pipe=None):
        """
        pickle_pipe : file name with path
        """
        self.pickle_pipe = pickle_pipe

        # extract_param
        self.extract_param = [
            "Frequency",
            "InstlmentMode",
            "LoanStatus",
            "PaymentMode",
            "BranchID",
            "Area",
            "Tenure",
            "AssetCost",
            "AmountFinance",
            "DisbursalAmount",
            "EMI",
            "DisbursalDate",
            "MaturityDAte",
            "AuthDate",
            "AssetID",
            "ManufacturerID",
            "SupplierID",
            "LTV",
            "SEX",
            "AGE",
            "MonthlyIncome",
            "City",
            "State",
            "ZiPCODE",
        ]

        # date
        self.date_parma = {"reference_date": "2020-12-31"}

        # type conversion
        self.dtype_param = {
            "cat_cols": [
                "BranchID",
                "ManufacturerID",
                "SupplierID",
                "ZiPCODE",
                "Frequency",
                "InstlmentMode",
                "LoanStatus",
                "PaymentMode",
                "Area",
                "SEX",
                "City",
                "State",
                "AuthDate_day",
                "AuthDate_month",
                "AuthDate_weekday",
                "AuthDate_year",
                "MaturityDAte_day",
                "MaturityDAte_month",
                "MaturityDAte_weekday",
                "MaturityDAte_year",
            ],
            "num_cols": [
                "disbursa_months",
                "Tenure",
                "AssetCost",
                "AmountFinance",
                "DisbursalAmount",
                "EMI",
                "LTV",
                "AGE",
                "MonthlyIncome",
            ],
            "date_cols": [],
        }

        # outlier_Params
        self.outlier_Params = {
            "new_outlier_col": True,
            "cols": [
                "disbursa_months",
                "Tenure",
                "AssetCost",
                "AmountFinance",
                "DisbursalAmount",
                "EMI",
                "LTV",
                "AGE",
                "MonthlyIncome",
            ],
        }

        # impute_param
        self.impute_param = {
            "cat_cols": [
                "BranchID",
                "ManufacturerID",
                "Frequency",
                "InstlmentMode",
                "LoanStatus",
                "PaymentMode",
                "Area",
                "SEX",
                "City",
                "State",
                "AuthDate_day",
                "AuthDate_month",
                "AuthDate_weekday",
                "AuthDate_year",
                "MaturityDAte_day",
                "MaturityDAte_month",
                "MaturityDAte_weekday",
                "MaturityDAte_year",
            ],
            "num_cols": [
                "disbursa_months",
                "Tenure",
                "AssetCost",
                "AmountFinance",
                "DisbursalAmount",
                "EMI",
                "LTV",
                "AGE",
                "MonthlyIncome",
            ],
            "date_cols": ["DisbursalDate", "MaturityDAte", "AuthDate"],
            "dont_use_cols": ["AssetID", "ZiPCODE", "SupplierID"],
            "n_estimators": 50,
            "max_depth": 8,
        }

        # dirchlet encoder
        self.dirichlet_param = {
            "cat_cols": {
                "BranchID": "BranchID",
                "Area": "Area",
                "SupplierID": "SupplierID",
                "ZiPCODE": "ZiPCODE",
                "Place": ["City", "State"],
            },
            "moments": "mean",
            "n_samples": 50,
            "sample_size": 0.8,
            "random_state": seed,
        }

        # quantile transform
        self.quant_param = {
            "cols": [
                "disbursa_months",
                "Tenure",
                "AssetCost",
                "AmountFinance",
                "DisbursalAmount",
                "EMI",
                "LTV",
                "AGE",
                "MonthlyIncome",
            ],
            "n_quantiles": 5000,
            "output_distribution": "normal",
            "random_state": seed,
        }

        self.dummy_param = [
            "AuthDate_day",
            "AuthDate_month",
            "AuthDate_weekday",
            "AuthDate_year",
            "MaturityDAte_day",
            "MaturityDAte_month",
            "MaturityDAte_weekday",
            "MaturityDAte_year",
            "ManufacturerID",
            "Frequency",
            "InstlmentMode",
            "LoanStatus",
            "PaymentMode",
            "SEX",
        ]
        self.drop_columns = [
            "DisbursalDate",
            "MaturityDAte",
            "AuthDate",
            "AuthDate_day",
            "AuthDate_month",
            "AuthDate_weekday",
            "AuthDate_year",
            "MaturityDAte_day",
            "MaturityDAte_month",
            "MaturityDAte_weekday",
            "MaturityDAte_year",
            "ManufacturerID",
            "Frequency",
            "InstlmentMode",
            "LoanStatus",
            "PaymentMode",
            "SEX",
            "AssetID",
            "BranchID",
            "SupplierID",
            "Area",
            "City",
            "State",
            "ZiPCODE",
        ]

    def feature_pipe(self):
        """
        # Feature Engineering Pipeline
        # Step: ExtractColumn
        # Step: date Preprocessing
        # Step: Impute
        # Step: Dirichlet
        # Step: QuantileTransformer
        # Step: DummyVariable
        # Step: Drop Columns1
        """

        print("Feature Engineering Pipeline", "-" * 20)
        pipe = Pipeline(
            [
                ("ExtractColumn", custom.ExtractColumn(self.extract_param)),
                ("preprocess", preprocess.Preprocess()),
                ("date", feat_engineer.DateTransform(**self.date_parma)),
                ("handle outliers", preprocess.HandleOutlier(**self.outlier_Params)),
                ("TypeConversion", preprocess.TypeConversion(**self.dtype_param)),
                ("Impute", custom_impute.MissForestImputer(**self.impute_param)),
                ("Dirichlet", encoder.DirichletEncoder(**self.dirichlet_param)),
                ("DummyVariable", feat_engineer.DummyTransformer(self.dummy_param)),
                ("demography", demography.DemographyFeat()),
                ("Quan", preprocess.CustomQuantileTransformer(**self.quant_param)),
                ("Drop Columns1", custom.DropColumn(self.drop_columns)),
            ]
        )

        return pipe

    def fit_y(self, y):
        le_target = LabelEncoder()
        le_target.fit(y.values.ravel())
        joblib.dump(le_target, f"models/{self.pickle_pipe}_y.pkl")
        if isinstance(y, pd.Series):
            y = le_target.transform(y)
            y = pd.Series(y)
        else:
            y = le_target.transform(y)
            y = y.reshape(-1, 1)
        return y

    def transform_y(self, y):
        le_target = joblib.load(le_target, f"models/{self.pickle_pipe}_y.pkl")
        y = le_target.inverse_transform(y)
        return y

    def fit_transform_pipe(self, X, y=None):
        self.pipe_X = self.feature_pipe()

        y = self.fit_y(y)

        # fit transform
        X = self.pipe_X.fit_transform(X, y)
        if isinstance(self.pickle_pipe, str):
            joblib.dump(self.pipe_X, f"models/{self.pickle_pipe}_X.pkl")
        return X, y

    def transform_pipe(self, X, y=None):
        X = self.pipe_X.transform(X)
        target_cols = ["Top-up Month"]
        # y = self.transform_y(y, target_cols)

        return X, y


class CustomPipelineBureau:
    """
    Custom Pipeline
    """

    def __init__(self, pickle_pipe=None):
        """
        pickle_pipe : file name with path
        """
        self.pickle_pipe = pickle_pipe

        # extract_param
        self.extract_param = [
            "ID",
            "SELF-INDICATOR",
            "MATCH-TYPE",
            "ACCT-TYPE",
            "CONTRIBUTOR-TYPE",
            "DATE-REPORTED",
            "OWNERSHIP-IND",
            "ACCOUNT-STATUS",
            "DISBURSED-DT",
            "CLOSE-DT",
            "LAST-PAYMENT-DATE",
            "CREDIT-LIMIT/SANC AMT",
            "DISBURSED-AMT/HIGH CREDIT",
            "INSTALLMENT-AMT",
            "CURRENT-BAL",
            "INSTALLMENT-FREQUENCY",
            "OVERDUE-AMT",
            "WRITE-OFF-AMT",
            "ASSET_CLASS",
            "REPORTED DATE - HIST",
            "DPD - HIST",
            "CUR BAL - HIST",
            "AMT OVERDUE - HIST",
            "AMT PAID - HIST",
            "TENURE",
        ]

        # type conversion
        self.dtype_param = {
            "cat_cols": [
                "MATCH-TYPE",
                "ACCT-TYPE",
                "CONTRIBUTOR-TYPE",
                "ACCOUNT-STATUS",
                "OWNERSHIP-IND",
                "ASSET_CLASS",
                "INSTALLMENT-FREQUENCY",
            ],
            "num_cols": [],
            "date_cols": [],
        }

        # outlier_Params
        self.outlier_Params = {
            "new_outlier_col": True,
            "cols": [
                "INSTALLMENT_AMT__mean",
                "INSTALLMENT_AMT__std",
                "CREDIT_LIMIT_SANC_AMT__mean",
                "CREDIT_LIMIT_SANC_AMT__std",
                "DISBURSED_AMT_HIGH_CREDIT__mean",
                "DISBURSED_AMT_HIGH_CREDIT__std",
                "CURRENT_BAL__mean",
                "CURRENT_BAL__std",
                "OVERDUE_AMT__mean",
                "OVERDUE_AMT__std",
                "WRITE_OFF_AMT__mean",
                "WRITE_OFF_AMT__std",
                "TENURE__mean",
                "TENURE__std",
                "AMT_PAID_HIST_sum__mean",
                "AMT_PAID_HIST_sum__std",
                "AMT_OVERDUE_HIST_sum__mean",
                "AMT_OVERDUE_HIST_sum__std",
                "CUR_BAL_HIST_sum__mean",
                "CUR_BAL_HIST_sum__std",
            ],
        }

        # impute_param
        self.impute_param = {
            "cat_cols": [],
            "num_cols": [],
            "date_cols": [],
            "dont_use_cols": ["AssetID", "SupplierID", "ZiPCODE"],
            "n_estimators": 50,
            "max_depth": 6,
        }

        # pca
        self.pca_param = {
            "n_components": 4,
            "cols": [],
            "col_prefix": "pca",
            "seed": seed,
        }

        # quantile transform
        self.quant_param = {
            "cols": [
                "INSTALLMENT_AMT__mean",
                "INSTALLMENT_AMT__std",
                "CREDIT_LIMIT_SANC_AMT__mean",
                "CREDIT_LIMIT_SANC_AMT__std",
                "DISBURSED_AMT_HIGH_CREDIT__mean",
                "DISBURSED_AMT_HIGH_CREDIT__std",
                "CURRENT_BAL__mean",
                "CURRENT_BAL__std",
                "OVERDUE_AMT__mean",
                "OVERDUE_AMT__std",
                "WRITE_OFF_AMT__mean",
                "WRITE_OFF_AMT__std",
                "TENURE__mean",
                "TENURE__std",
                "AMT_PAID_HIST_sum__mean",
                "AMT_PAID_HIST_sum__std",
                "AMT_OVERDUE_HIST_sum__mean",
                "AMT_OVERDUE_HIST_sum__std",
                "CUR_BAL_HIST_sum__mean",
                "CUR_BAL_HIST_sum__std",
            ],
            "n_quantiles": 5000,
            "output_distribution": "normal",
            "random_state": seed,
        }

        self.dummy_param = []
        self.drop_columns = []

    def feature_pipe(self):
        """
        # Feature Engineering Pipeline
        # Step: ExtractColumn
        # Step: Data Preprocessing
        # Step: HandleOutlier
        # Step: QuantileTransformer
        """

        print("Feature Engineering Pipeline", "-" * 20)
        pipe = Pipeline(
            [
                ("ExtractColumn", custom.ExtractColumn(self.extract_param)),
                ("TypeConversion", preprocess.TypeConversion(**self.dtype_param)),
                ("Burea feat", bureau_feat.BureauFeat()),
                # ("handle outliers", preprocess.HandleOutlier(**self.outlier_Params)),
                ("Quan", preprocess.CustomQuantileTransformer(**self.quant_param)),
                # ("Drop Columns1", custom.DropColumn(self.drop_columns)),
            ]
        )

        return pipe

    def fit_transform_pipe(self, X, y=None):
        self.pipe_X = self.feature_pipe()

        # fit transform
        X = self.pipe_X.fit_transform(X, y)
        if isinstance(self.pickle_pipe, str):
            joblib.dump(self.pipe_X, f"models/{self.pickle_pipe}_X.pkl")
        return X, y

    def transform_pipe(self, X, y=None):
        X = self.pipe_X.transform(X)
        # target_cols = ["Top-up Month"]
        # y = self.transform_y(y, target_cols)

        return X, y