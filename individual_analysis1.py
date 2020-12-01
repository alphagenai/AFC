# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:47:04 2020

@author: mark
"""


import os
import pandas as pd
from google.cloud import bigquery
import random
import matplotlib.pyplot as plt

#from google.oauth2 import service_account


key_path = r"C:\Users\mat4m_000\Documents\Wellow data\SFC\AFCproj-keyfile.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
#client = bigquery.Client()


SQL = """ 
    Select p.TransactionTS,
        p.AmountPaid,
        p.ContractId
    FROM afcproj.files_dupe.Payments_2020_11_17 p
    inner join afcproj.files_dupe.jan_19_cohort j
        on p.ContractId = j.ContractId    
    WHERE p.paymentStatusTypeEntity != 'REFUSED'
        and
        p.PaymentResultTypeEntity != 'PAYMENT_FREE_CREDIT'

    UNION ALL
    Select a.CreatedAt,
        a.Amount,
        a.ContractId
    FROM afcproj.files_dupe.Adjustments_2020_11_17 a
    inner join afcproj.files_dupe.jan_19_cohort j
        on a.ContractId = j.ContractId
    WHERE a.BalanceChangeType = 'MANUAL'

        """

#for contractID in cohort:
df = pd.read_gbq(SQL,) #chunksize=10000) #chunksize doesnt work

df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S')

df = df.set_index(['ContractId','TransactionTS'])
              
df = df.astype('float64')


MEAN_PAYMENT = df.mean()


hundred_random_IDs = random.sample(df.index.get_level_values(0).unique().values.tolist(), k=100)

small_df = df.loc[hundred_random_IDs]   # see which IDs --> small_df.index.get_level_values(0).unique()

sdf = small_df['AmountPaid'].unstack(0).fillna(0).sort_index()
cum_df = sdf.cumsum(axis=0)

ax = cum_df.plot(figsize=(20,8), legend=False, title="Cumulative Payments for 100 Random Contracts in Jan 2019 Cohort")
ax.set_xlabel("Transaction Date")
ax.set_ylabel("Total Repayment Amount")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000)+'k'))
plt.savefig('Cumulative Payments for 100 Random Contracts in Jan 2019 Cohort')

daily_sdf = sdf.groupby(sdf.index.date).sum() #all payments in one day are grouped together
daily_cum_sdf = daily_sdf.cumsum(axis=0) 

timeseries_length = (daily_sdf.index.max() - daily_sdf.index.min()).days

AVERAGE_PAYMENT_FREQUENCY = timeseries_length / daily_sdf.astype(bool).sum(axis=0) 
MEAN_PAYMENT_PER_DAY = daily_sdf.sum()/ timeseries_length

## HISTAGRAM OF daily % rePAYMENTS

