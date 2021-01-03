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
monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)


PD_given_D, PD_given_ND, PND_given_D, PND_given_ND = calc_PD(monthly_sdf_pivot, )