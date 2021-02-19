# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:22:53 2021

@author: mark
"""

import pandas as pd

from individual_analysis1 import create_percent_sdf


cohort='dec_17'

df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
df = df.groupby(['ContractId', 'TransactionTS']).sum()


MONTHLY_SDF = monthly_sdf = df.groupby(['ContractId', pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
MONTHLY_SDF_PIVOT = monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
MONTHLY_PERCENT_PIVOT = monthly_percent_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=False, cohort='dec_17')

DAILY_SDF = daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='D', level=1)])['AmountPaid'].sum()
DAILY_SDF_PIVOT = daily_sdf_pivot = daily_sdf.unstack(0).fillna(0)

DAILY_CUMULATIVE_PERCENT_SDF_PIVOT = daily_cumulative_percent_sdf_pivot = create_percent_sdf(daily_sdf_pivot, cumulative=True, cohort=cohort)
MONTHLY_CUMULATIVE_PERCENT_SDF_PIVOT = monthly_cumulative_percent_sdf_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=True, cohort=cohort)

DAILY_FULLY_PAID = daily_fully_paid = daily_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag
MONTHLY_FULLY_PAID = monthly_fully_paid = monthly_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag


MONTHLY_UNFINISHED_CONTRACTS = monthly_percent_pivot.mask(monthly_fully_paid)
