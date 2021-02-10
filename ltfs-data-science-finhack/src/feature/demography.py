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
from sklearn import preprocessing


class DemographyFeat(BaseEstimator, TransformerMixin):
    def __init__(self):
        # super().__init__()
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # demograpy feature
        X["monthly_surplus"] = X["EMI"] - X["MonthlyIncome"]
        X["EMI_to_income_ratio"] = X["EMI"] / X["MonthlyIncome"]
        X["Total_amount_to_pay"] = X["EMI"] * X["Tenure"]
        X["Total_income_within_loan"] = X["MonthlyIncome"] * X["Tenure"]
        X["Interest_amount"] = X["Total_amount_to_pay"] - X["DisbursalAmount"]

        X["Residual_amount"] = X["Total_income_within_loan"] - X["Total_amount_to_pay"]
        X["Asset_to_total_income"] = X["AssetCost"] / X["Total_income_within_loan"]
        X["Asset_to_monthly_income"] = X["AssetCost"] / X["MonthlyIncome"]
        X["pay_to_loan_ratio"] = X["Total_amount_to_pay"] / X["DisbursalAmount"]

        X["Finance_to_disbursed_ratio"] = X["DisbursalAmount"] / X["AmountFinance"]

        X["finance_leverage_money"] = X["AssetCost"] - X["AmountFinance"]
        X["finance_leverage_ratio"] = X["AssetCost"] / X["AmountFinance"]

        # X["disbursal_leverage_money"] = X["AssetCost"] - X["DisbursalAmount"]
        # X["disbursal_leverage_ratio"] = X["AssetCost"] / X["DisbursalAmount"]
        # X["load_disbursal_ratio"] = X["DisbursalAmount"] / X["Tenure"]

        X["processing_fees"] = X["AmountFinance"] - X["DisbursalAmount"]

        return X
