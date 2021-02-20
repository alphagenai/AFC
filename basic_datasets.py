# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:22:53 2021

@author: mark
"""

import pandas as pd

from individual_analysis1 import create_percent_sdf
from calculate_days_dropped import calculate_days_dropped

class BasicDatasets(object):
    def __init__(self, cohort='dec_17'):
        self.cohort = cohort
        df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
        df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
        self.df = df.groupby(['ContractId', 'TransactionTS']).sum()
        
    @property
    def monthly_sdf(self):
        return self.df.groupby(['ContractId', pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
    
    @property
    def monthly_sdf_pivot(self):
        return self.monthly_sdf.unstack(0).fillna(0)

    @property
    def monthly_percent_pivot(self):
        return create_percent_sdf(self.monthly_sdf_pivot, cumulative=False, cohort=self.cohort)


    @property
    def monthly_cumulative_percent_sdf_pivot(self):
        return create_percent_sdf(self.monthly_sdf_pivot, cumulative=True, cohort=self.cohort)


    @property
    def daily_sdf(self):
        return self.df.groupby(['ContractId', pd.Grouper(freq='D', level=1)])['AmountPaid'].sum()
    
    @property
    def daily_sdf_pivot(self):
        return self.daily_sdf.unstack(0).fillna(0)

    @property
    def daily_percent_pivot(self):
        return create_percent_sdf(self.daily_sdf_pivot, cumulative=False, cohort=self.cohort)

    @property
    def daily_cumulative_percent_sdf_pivot(self): 
        return create_percent_sdf(self.daily_sdf_pivot, cumulative=True, cohort=cohort)

    @property
    def daily_full_ts(self):
        return calculate_days_dropped(self.df.groupby(['ContractId', pd.Grouper(freq='M', level=1)]).sum())

cohort='dec_17'
bds = BasicDatasets()
df = bds.df
MONTHLY_SDF = monthly_sdf = df.groupby(['ContractId', pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
MONTHLY_SDF_PIVOT = monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
MONTHLY_PERCENT_PIVOT = monthly_percent_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=False, cohort=cohort)

DAILY_SDF = daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='D', level=1)])['AmountPaid'].sum()
DAILY_SDF_PIVOT = daily_sdf_pivot = daily_sdf.unstack(0).fillna(0)

DAILY_CUMULATIVE_PERCENT_SDF_PIVOT = daily_cumulative_percent_sdf_pivot = create_percent_sdf(daily_sdf_pivot, cumulative=True, cohort=cohort)
MONTHLY_CUMULATIVE_PERCENT_SDF_PIVOT = monthly_cumulative_percent_sdf_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=True, cohort=cohort)

DAILY_FULLY_PAID = daily_fully_paid = daily_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag
MONTHLY_FULLY_PAID = monthly_fully_paid = monthly_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag


MONTHLY_UNFINISHED_CONTRACTS = monthly_percent_pivot.mask(monthly_fully_paid)
