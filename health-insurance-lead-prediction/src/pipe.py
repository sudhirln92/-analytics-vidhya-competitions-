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
from sklearn.pipeline import Pipeline
import joblib

from .preprocess import preprocess, custom
from .preprocess import custom_impute
from .feature import encoder, feat_engineer
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
            "City_Code",
            "Region_Code",
            "Accomodation_Type",
            "Reco_Insurance_Type",
            "Is_Spouse",
            "Health Indicator",
            "Holding_Policy_Type",
            "Reco_Policy_Cat",
            "Holding_Policy_Duration",
            "Upper_Age",
            "Lower_Age",
            "Reco_Policy_Premium",
        ]

        cat_cols = [
            "City_Code",
            "Region_Code",
            "Accomodation_Type",
            "Reco_Insurance_Type",
            "Is_Spouse",
            "Health Indicator",
            "Holding_Policy_Type",
            "Reco_Policy_Cat",
            "Holding_Policy_Duration",
        ]
        num_cols = ["Upper_Age", "Lower_Age", "Reco_Policy_Premium"]

        # type conversion
        self.dtype_param = {
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "date_cols": [],
        }

        # second level combination
        self.combine_cat_param = {
            "columns": [
                "City_Code",
                "Region_Code",
                "Accomodation_Type",
                "Reco_Insurance_Type",
                "Is_Spouse",
                "Health Indicator",
                "Holding_Policy_Type",
                "Reco_Policy_Cat",
                "Holding_Policy_Duration",
            ]
        }
        self.bin_param = {
            "bins": {
                "Reco_Policy_Premium": [
                    0,
                    1000,
                    5000,
                    10000,
                    20000,
                    25000,
                    30000,
                    100000,
                ],
                "Lower_Age": [0, 20, 30, 40, 50, 60, 100],
                "Upper_Age": [0, 20, 30, 40, 50, 60, 100],
            }
        }

        # dirchlet encoder
        self.beta_param = {
            "cat_cols": {
                "Region_Code": "Region_Code",
            },
            "moments": "mean",
            "n_samples": 50,
            "sample_size": 0.8,
            "random_state": seed,
        }

        # quantile transform
        self.quant_param = {
            "cols": [
                "Upper_Age",
                "Lower_Age",
                "premium_to_age1",
                "premium_to_age2",
                "duration",
                "Reco_Policy_Premium",
            ],
            "n_quantiles": 1000,
            "output_distribution": "normal",
            "random_state": seed,
        }

        self.dummy_param = [
            "City_Code",
            "Accomodation_Type",
            "Reco_Insurance_Type",
            "Is_Spouse",
            "Health Indicator",
            "Holding_Policy_Type",
            "Reco_Policy_Cat",
            "Holding_Policy_Duration",
        ]
        self.drop_columns = [
            "City_Code",
            "Accomodation_Type",
            "Reco_Insurance_Type",
            "Is_Spouse",
            "Health Indicator",
            "Holding_Policy_Type",
            "Reco_Policy_Cat",
            "Holding_Policy_Duration",
            "Region_Code",
        ]

    def feature_pipe(self):
        """
        # Feature Engineering Pipeline
        # ExtractColumn
        # date Preprocessing
        # Impute
        # beta
        # Ratio Feat
        # Bin Feat
        # QuantileTransformer
        # DummyVariable
        # Drop Columns1
        """

        print("Feature Engineering Pipeline", "-" * 20)
        pipe = Pipeline(
            [
                ("ExtractColumn", custom.ExtractColumn(self.extract_param)),
                ("Impute", custom_impute.SimpleImpute()),
                ("TypeConversion", preprocess.TypeConversion(**self.dtype_param)),
                ("combine cat", feat_engineer.CombineCatFeat(**self.combine_cat_param)),
                ("beta", encoder.BetaEncoder(**self.beta_param)),
                ("RatioFeat", feat_engineer.RatioFeat()),
                ("BinFeat", feat_engineer.BinFeat(**self.bin_param)),
                ("DummyVariable", feat_engineer.DummyTransformer(self.dummy_param)),
                ("Quan", preprocess.CustomQuantileTransformer(**self.quant_param)),
                ("Drop Columns", custom.DropColumn(self.drop_columns)),
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
        # y = self.transform_y(y, target_cols)

        return X, y
