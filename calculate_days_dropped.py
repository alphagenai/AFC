# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:23:07 2020

@author: mark
"""


daily_sdf = small_df.groupby(['ContractId', pd.Grouper(freq='D', level=1)]).sum()


daily_sdf['TransactionTS'] = daily_sdf.index.get_level_values(1)

## unstack here?
#daily_sdf['prev_payment_date'] = daily_sdf['TransactionTS'].shift(1)

daily_sdf[['prev_payment_date', 'prev_duration']] = daily_sdf.groupby(level=0)[['TransactionTS', 'Duration']].shift(1)

## days since token dropped
# WHAT HAPPENS TO ADJUSTMENTS
daily_sdf['days_dropped'] = daily_sdf['TransactionTS']  - daily_sdf['prev_payment_date'] - pd.to_timedelta(daily_sdf['prev_duration'], unit='D')


