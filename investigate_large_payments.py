# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 17:59:18 2021

@author: mark
"""

import pandas as pd

from individual_analysis1 import create_small_df, create_percent_sdf, convert_to_daily_pivot

## investigate large payments

df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
df = df.groupby(['ContractId', 'TransactionTS']).sum()

monthly_sdf = df.groupby(['ContractId', pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)

monthly_percent_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=False, cohort='dec_17')

high_payments = monthly_percent_pivot.loc['2018-01-31':, :] > 0.2

high_payments.sum().sum()

monthly_percent_pivot.loc[:, high_payments.any(axis=0)]

df.loc['1350678']