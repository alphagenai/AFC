# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:27:40 2020

@author: mark
"""

import pandas as pd
import numpy as np

from calc_PD import calc_PD

df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)

df = df.groupby(['ContractId', 'TransactionTS']).sum()

daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
monthly_sdf = daily_sdf.groupby(['ContractId',pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
monthly_sdf_pivot = monthly_sdf['AmountPaid'].unstack(0).fillna(0)


PD_given_D, PD_given_ND, PND_given_D, PND_given_ND = calc_PD(monthly_sdf_pivot,)
monthly_sdf_fullts = calc_moving_average(monthly_sdf)

ma_pivot = monthly_sdf_fullts['MovingAverage'].unstack(0).shift(1).fillna(method='ffill') #shift(1) for next month forecast, ffill for future months with no MA (because no payments made)

## logic

"""

if month = paid:
    month+1 = paid*PNDgivenND + PDgivenND*0
elif month != paid
    month+1 = paid*PNDgivenD + PDgivenD*0
    
"""

## SHOULD  PND_given_D + PND_given_ND = 1?

one_month_forecast_if_paid = ma_pivot.multiply(PND_given_ND)
one_month_forecast_if_default = ma_pivot.multiply(PND_given_D)


one_contract_id = monthly_sdf_pivot.columns[3]

one_ma = ma_pivot[one_contract_id]
forecast_date = '2019-12-31'

defaults = monthly_sdf_pivot.iloc[1:]==0 # total non-NaN: 28,593

if defaults.loc[forecast_date, one_contract_id]:
    