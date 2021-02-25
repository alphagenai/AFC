# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:35:54 2021

@author: mark
"""


# 1. days of not paying against final Value
# 2. average payment against final value
import pandas as pd
import numpy as np

from basic_datasets import BasicDatasets

bd = BasicDatasets()

dcpp = bds.daily_cumulative_percent_sdf_pivot
dpp = bds.daily_percent_pivot.mask(bds.daily_fully_paid)

dfts = pd.merge(bds.daily_full_ts,
    dcpp.stack().swaplevel().to_frame('cum_val'),
    left_index=True,
    right_index=True)


ultimate_loss = (1 - dcpp.iloc[-1]).clip(0,1).to_frame('Ulitimate_loss')

days_in_ts = (dfts.index.get_level_values(1).max() - dfts.index.get_level_values(1).min()).days

lookback_days_out = {}
lookback_average_pay = {}
idx = pd.IndexSlice

#data_df = dfts['days_out_of_elec'].loc[idx[:,'Jan-2018':'Oct-2020'],].unstack(1)
df = dfts[dfts['cum_val']<0.99].loc[idx[:,'Jan-2018':'Oct-2020'],]['days_out_of_elec']


#df = data_df.join(ultimate_loss, )

# for i in range(start=30, stop=days_in_ts, step=10):
#     column_name = 'Lookback {}'.format(i)
#     lookback_days_out[column_name] = dfts['days_out_of_elec'].clip(upper=i)
#     #lookback_average_pay[column_name] = dfts['AmountPaid'].rolling(i).mean()
    
#df = pd.DataFrame.from_dict(lookback_days_out)

#dfts['days_out_of_elec'].clip(upper=i).reset_index().join(ultimate_loss, on='ContractId').groupby(['days_out_of_elec', 'TransactionTS']).mean()

bins = [(x-1)/10000. for x in range(21)] + [(x+2)/1000. for x in range(8)] + [(x+1)/100. for x in range(10)]

d_dict = {}
a_dict = {}

d_count_dict = {}
a_count_dict = {}

for i in range(10, np.min([days_in_ts, 190]), 10):
    dg = (df.clip(upper=i) == i).to_frame('lookback_{}'.format(i)).join(ultimate_loss).groupby(['lookback_{}'.format(i), 'TransactionTS'])
    rolling_mean_df = pd.cut(dpp.rolling(i).mean().stack().clip(0,1), bins=bins, ).to_frame('lookback_{}'.format(i)).join(ultimate_loss)
    ag = rolling_mean_df.groupby(['TransactionTS', 'lookback_{}'.format(i)])
    # optional
    # dg.count().unstack(0) # for debugging
    # ag.count().unstack(1) 

    d_dict[i] = dg.mean().unstack(0)
    a_dict[i] = ag.mean().unstack(1).dropna(axis=1, how='all',)

    d_count_dict[i] = dg.count().unstack(0)
    a_count_dict[i] = ag.count().unstack(1).dropna(axis=1, how='all',)


pd.concat(d_dict.values(), axis=1, keys=d_dict.keys()).to_csv('files\\default_lookup_table.csv')
pd.concat(a_dict.values(), axis=1, keys=a_dict.keys()).to_csv('files\\average_payment_table.csv')

pd.concat(d_count_dict.values(), axis=1, keys=d_count_dict.keys()).to_csv('files\\default_counts.csv')
pd.concat(a_count_dict.values(), axis=1, keys=a_dict.keys()).to_csv('files\\average_payment_counts.csv')


""" 
    EAD - amount left to pay at default
    
    amount of time into the contract that default occurs

    look into the wierd transactions that happen on the last day of the month

"""