# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:47:04 2020

@author: mark
"""


import os
import pandas as pd
from google.cloud import bigquery
#from google.oauth2 import service_account


key_path = r"C:\Users\mat4m_000\Documents\Wellow data\SFC\AFCproj-keyfile.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
client = bigquery.Client()


SQL = """ 
    Select p.TransactionTS,
        p.AmountPaid,
        p.ContractId
    FROM afcproj.files_dupe.Payments_2020_11_17 p
    inner join afcproj.files_dupe.jan_19_cohort j
        on p.ContractId = j.ContractId    
    WHERE j.ContractId in (
        '1555471',
        '1569214',
        '1557804'
        )

    UNION ALL
    Select a.CreatedAt,
        a.Amount,
        a.ContractId
    FROM afcproj.files_dupe.Adjustments_2020_11_17 a
    inner join afcproj.files_dupe.jan_19_cohort j
        on a.ContractId = j.ContractId
    WHERE j.ContractId in (
        '1555471',
        '1569214',
        '1557804'
        )
        """

df = pd.read_gbq(SQL, chunksize=10000)

df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S')

df = df.set_index(['ContractId','TransactionTS'])
              
df = df.astype('float64')
df.loc[['1568884',
    '1570013',
    '1571049']].unstack(0).plot()

df = df['AmountPaid'].fillna(0).sort_index()
df.cumsum(axis=0).plot()
df.groupby(df.index.date).sum().cumsum(axis=0).plot()  #all payments in one day are grouped together


AVERAGE_PAYMENT_FREQUENCY = (df.index.max() - df.index.min()).days / df.astype(bool).sum(axis=0) 
