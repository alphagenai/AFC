# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:54:53 2021

@author: mark
"""


import pandas as pd

from individual_analysis1 import create_percent_sdf


df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)

monthly_sdf = df.groupby(['ContractId',pd.Grouper(key='TransactionTS', freq='M',)])['AmountPaid'].sum()

monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)

monthly_perc_sdf_pivot = create_percent_sdf(monthly_sdf_pivot,
                                            cumulative=False, use_monthdiff=False, cohort='dec_17')


monthly_diff = monthly_perc_sdf_pivot.diff()
monthly_diff.plot(legend=False)

monthly_diff.mean(axis=1)


