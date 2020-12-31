# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:58:14 2020

@author: mark
"""




import pandas as pd
import numpy as np

from individual_analysis1 import create_small_df, create_percent_sdf, convert_to_daily_pivot


df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)

df = df.groupby(['ContractId', 'TransactionTS']).sum()

daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
daily_sdf_pivot = daily_sdf.unstack(0).fillna(0)
daily_percent_pivot = create_percent_sdf(daily_sdf_pivot, cumulative=False, use_monthdiff=False, cohort='dec_17')


monthly_sdf = daily_sdf.groupby(['ContractId',pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()

monthly_percent_pivot = daily_percent_pivot.groupby(pd.Grouper(freq='M')).sum()
