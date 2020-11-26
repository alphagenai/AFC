# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:05:50 2020

@author: mark
"""



import os
import pandas as pd
from google.cloud import bigquery
import numpy as np

## It is necessary to create your Google Authentication keys before accessing BigQuery via Python
key_path = r"C:\Users\mat4m_000\Documents\Wellow data\SFC\AFCproj-keyfile.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

client = bigquery.Client()

SQL = """
SELECT
    SUM(total_contract_value) as sum_total_contract_value,
    SUM(AmountPaid) as sum_amount_paid,
    monthdiff,
    EXTRACT(YEAR FROM RegistrationDate) AS cohort_year,
    EXTRACT(MONTH FROM RegistrationDate) AS cohort_month,

FROM
    ( 
    
    SELECT 
        p.TransactionTs,
        c.RegistrationDate,       
        datetime_diff(DATETIME(p.TransactionTs), c.RegistrationDate, MONTH) as monthdiff,
        p.AmountPaid
    FROM `afcproj.files_dupe.Payments_2020_11_17` p
    JOIN `afcproj.files_dupe.Contracts_20201117` c
        on p.ContractId = c.ContractId 
    WHERE 
        p.paymentStatusTypeEntity != 'REFUSED'
        and
        p.PaymentResultTypeEntity != 'PAYMENT_FREE_CREDIT'
        and c.PaymentMethod = 'FINANCED'
    )
    
GROUP BY monthdiff, cohort_year, cohort_month

"""

df = pd.read_gbq(SQL,)
df['day'] = 1

df.index = pd.to_datetime(
    df[['cohort_year', 'cohort_month', 'day']].rename(
                 columns={'cohort_year':'year', 'cohort_month':'month',}
        )
    )

df2 = df[['sum_amount_paid', 'monthdiff']]
df3 = df2.pivot(columns=['monthdiff'],).astype('float64')
df4 = df3.cumsum(axis=1)
df4.to_csv('temp4.csv')
