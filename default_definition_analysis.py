# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:55:23 2021

@author: mark
"""


import pandas as pd

from individual_analysis1 import create_percent_sdf
from calculate_days_dropped import calculate_days_dropped




""" NEED TO REMOVE CONTRACTS THAT ARE FULLY PAID UP 
## tO DO - unstack and mask fully paid 3x, as before
"""
df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
daily_sdf = df.groupby(['ContractId',pd.Grouper(key='TransactionTS', freq='D',)]).sum()
daily_sdf_pivot = daily_sdf['AmountPaid'].unstack(0).fillna(0)

daily_cumulative_percent_sdf_pivot = create_percent_sdf(daily_sdf_pivot, cumulative=True, cohort='dec_17')
fully_paid = daily_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag

# daily_cumulative_percent_sdf = daily_cumulative_percent_sdf_pivot.stack().swaplevel(0,1)
# daily_cumulative_percent_sdf = daily_cumulative_percent_sdf_pivot.stack().reset_index().groupby(['ContractId','TransactionTS']).max()
# daily_cumulative_percent_sdf >=1.0

daily_sdf_fullts = calculate_days_dropped(daily_sdf)

idx = pd.IndexSlice
daily_sdf_fullts = daily_sdf_fullts.loc[idx[:,'2017-12-01':'2020-11-17'], :]

daily_sdf_fullts['PAR90+'] = (daily_sdf_fullts['days_out_of_elec'] > 90)

daily_sdf_fullts['par30_seqno'] = daily_sdf_fullts[daily_sdf_fullts['PAR30+']].groupby(
    ['ContractId','PAR30+']
    ).ngroup()

daily_sdf_fullts['TransactionTS'] = daily_sdf_fullts.index.get_level_values(1)

par30_sequence_end_date = daily_sdf_fullts.groupby(['ContractId', 'par30_seqno'])['TransactionTS'].max().mask(fully_paid.iloc[-1]).rename('end_date')
par30_sequence_start_date = daily_sdf_fullts.groupby(['ContractId', 'par30_seqno'])['TransactionTS'].min().mask(fully_paid.iloc[-1]).rename('start_date')

never_recover = pd.concat([par30_sequence_start_date,par30_sequence_end_date], axis=1, )[par30_sequence_end_date=='2020-11-17'].groupby('start_date').count().rename({'end_date':'never_recover'})
defaults_ts = daily_sdf_fullts[['PAR90+','PAR30+']].diff().reset_index().drop(columns='ContractId').groupby('TransactionTS').sum()

pd.concat([defaults_ts, never_recover ]).fillna(0)['Jan-2018':].plot()

## check big par30 spikes

p30_ts = daily_sdf_fullts['PAR30+'].unstack(0).diff(1).mask(fully_paid).sum(axis=1)
p90_ts = daily_sdf_fullts['PAR90+'].unstack(0).diff(1).mask(fully_paid).sum(axis=1)

pd.concat([p30_ts, p90_ts, never_recover], axis=1 ).rename({0:'PAR30+', 1:'PAR90+', 'end_date':'No Recovery'}, axis='columns').fillna(0).plot()