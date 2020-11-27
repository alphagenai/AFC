# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:05:50 2020

@author: mark
"""



import os
import pandas as pd
from google.cloud import bigquery
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

## It is necessary to create your Google Authentication keys before accessing BigQuery via Python
key_path = r"C:\Users\mat4m_000\Documents\Wellow data\SFC\AFCproj-keyfile.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

client = bigquery.Client()

## cohort payment timeseries
SQL = """
SELECT
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

pay_df = pd.read_gbq(SQL,)
pay_df['day'] = 1

pay_df.index = pd.to_datetime(
    pay_df[['cohort_year', 'cohort_month', 'day']].rename(
                 columns={'cohort_year':'year', 'cohort_month':'month',}
        )
    )

pay_df2 = pay_df[['sum_amount_paid', 'monthdiff']]



SQL = """
    
    SELECT
        SUM(AmountPaid) as sum_amount_paid,
        monthdiff,
        EXTRACT(YEAR FROM RegistrationDate) AS cohort_year,
        EXTRACT(MONTH FROM RegistrationDate) AS cohort_month,
    
    FROM
        ( 
        
        SELECT 
            a.CreatedAt,
            c.RegistrationDate,
            datetime_diff(DATETIME(a.CreatedAt), c.RegistrationDate, MONTH) as monthdiff,
            a.Amount as AmountPaid
        FROM `afcproj.files_dupe.Adjustments_2020_11_17` a
        JOIN `afcproj.files_dupe.Contracts_20201117` c
            on a.ContractId = c.ContractId 
        WHERE c.PaymentMethod = 'FINANCED'
        )
        
    GROUP BY monthdiff, cohort_year, cohort_month
    """

adj_df = pd.read_gbq(SQL,)
adj_df['day'] = 1

adj_df.index = pd.to_datetime(
    adj_df[['cohort_year', 'cohort_month', 'day']].rename(
                 columns={'cohort_year':'year', 'cohort_month':'month',}
        )
    )

adj_df2 = adj_df[['sum_amount_paid', 'monthdiff']]


payadj_df = pd.concat([pay_df2, adj_df2], axis=0)
payadj_df_grp = payadj_df.groupby([payadj_df.index, 'monthdiff']).sum() 

timeseries_df = payadj_df_grp.unstack('monthdiff').astype('float64')['sum_amount_paid']
cumsum_timeseries = timeseries_df.cumsum(axis=1)
cumsum_timeseries.sort_index().to_csv('temp.csv')


## total contract amount for each cohort
sql = """
SELECT 
    EXTRACT(YEAR FROM RegistrationDate) AS cohort_year,
    EXTRACT(MONTH FROM RegistrationDate) AS cohort_month,
    --SUM(Deposit) AS Deposit,  
    --SUM(Price) as Price,
    --SUM(AdditionalFee) as AdditionalFee,

    SUM(Deposit) + SUM(Price) + SUM(AdditionalFee) as TotalContractValue

FROM `afcproj.files_dupe.Contracts_20201117`
WHERE PaymentMethod = 'FINANCED'
GROUP BY cohort_month, cohort_year
"""


cohort_contract_sum_df = pd.read_gbq(sql,)
cohort_contract_sum_df['day'] = 1

cohort_contract_sum_df.index = pd.to_datetime(
    cohort_contract_sum_df[['cohort_year', 'cohort_month', 'day']].rename(
                 columns={'cohort_year':'year', 'cohort_month':'month',}
        )
    )
cohort_contract_sum_df.sort_index(inplace=True)

full_timeseries_df = pd.merge(
    cohort_contract_sum_df['TotalContractValue'].to_frame(),
    df4,
    left_index=True,
    right_index=True)

amort_timeseries_df = -df4.add(cohort_contract_sum_df['TotalContractValue'], axis=0)
amort_timeseries_df.to_csv('temp7.csv')

## dafuq is happening here??? how can adding two negatives give the right answer?
full_df = -df4.add(-cohort_contract_sum_df['TotalContractValue'], axis='index')

full_df.to_csv('temp8.csv')
df_to_plot = full_df['2017-06-01':]
df_to_plot.index = df_to_plot.index.month_name() + df_to_plot.index.year.astype('str')
ax = df_to_plot.T.loc[0:].plot(xticks=range(35), figsize=(20,8))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000000) + 'm'))
plt.savefig('amortization line chart.png')