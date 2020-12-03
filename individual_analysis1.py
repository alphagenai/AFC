# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:47:04 2020

@author: mark
"""


import google_sa_auth
import pandas as pd
from google.cloud import bigquery
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib as mpl
import matplotlib.ticker as ticker


#client = bigquery.Client()


SQL = """ 
    Select p.TransactionTS,
        p.AmountPaid,
        p.ContractId
    FROM afcproj.files_dupe.Payments_2020_11_17 p
    inner join afcproj.files_dupe.jan_19_cohort j
        on p.ContractId = j.ContractId   
    inner join afcproj.files_dupe.Contracts_20201117 c
        on c.ContractId = j.ContractId
    WHERE p.paymentStatusTypeEntity != 'REFUSED'
        and
        p.PaymentResultTypeEntity != 'PAYMENT_FREE_CREDIT'
        and c.Product = 'X850'
    UNION ALL
    Select a.CreatedAt,
        a.Amount,
        a.ContractId
    FROM afcproj.files_dupe.Adjustments_2020_11_17 a
    inner join afcproj.files_dupe.jan_19_cohort j
        on a.ContractId = j.ContractId
    inner join afcproj.files_dupe.Contracts_20201117 c
        on c.ContractId = j.ContractId

    WHERE a.BalanceChangeType = 'MANUAL'
        and c.Product = 'X850'

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




### Get Contract info

SQL = """
    SELECT c.ContractId,
        Price + AdditionalFee as TotalContractValue,     
        --c.RegistrationDate 
    FROM `afcproj.files_dupe.Contracts_20201117` c
    join `afcproj.files_dupe.jan_19_cohort` j
        on c.ContractId = j.ContractId
    """
cdf = pd.read_gbq(SQL,index_col='ContractId').astype('float64')


df = pd.merge(
    sdf.T,
    cdf,
    how='inner',
    left_index=True,
    right_index=True)

contract_values = df['TotalContractValue']

percent_df = df.divide(contract_values, axis=0).drop(columns=['TotalContractValue'])

cumulative_percent_sdf = percent_df.T.cumsum(axis=0)
cumulative_percent_sdf.index = pd.to_datetime(cumulative_percent_sdf.index,format='%Y/%m/%d %H:%M:%S')



daily_sdf = sdf.groupby(sdf.index.date).sum() #all payments in one day are grouped together
daily_cum_sdf = daily_sdf.cumsum(axis=0) 
monthly_sdf = sdf.groupby(pd.Grouper(freq='M')).sum()
monthly_cum_sdf = monthly_sdf.cumsum(axis=0)

timeseries_length = (daily_sdf.index.max() - daily_sdf.index.min()).days

AVERAGE_PAYMENT_FREQUENCY = timeseries_length / daily_sdf.astype(bool).sum(axis=0) 
MEAN_PAYMENT_PER_DAY = daily_sdf.sum()/ timeseries_length

###### Plotting
## Daily Payments
ax = daily_cum_sdf.plot(figsize=(20,8), legend=False, title="Cumulative Payments for 100 Random Contracts in Jan 2019 Cohort")
ax.set_xlabel("Transaction Date")
ax.set_ylabel("Total Repayment Amount")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000)+'k'))
plt.savefig('Cumulative Payments for 100 Random X850 Contracts in Jan 2019 Cohort')


## Smoothed Payments
daily_cum_sdf.rolling(window=30).mean().plot(figsize=(20,8), legend=False, title="Smoothed Cumulative Payments for 100 Random Contracts in Jan 2019 Cohort")
ax.set_xlabel("Transaction Date")
ax.set_ylabel("Total Repayment Amount")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000)+'k'))
plt.savefig('Smoothed Cumulative Payments for 100 Random  X850 Contracts in Jan 2019 Cohort')



## Percentage Payments
cumulative_percent_sdf.plot(figsize=(20,8), legend=False, title="Cumulative % Payments for 100 Random Contracts in Jan 2019 Cohort")
ax.set_xlabel("Transaction Date")
ax.set_ylabel("Total Repayment Amount % of Contract Value")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0%}'.format(x)))
plt.savefig('Cumulative percentage Payments for 100 Random Contracts in Jan 2019 Cohort')



## Monthly Percentage Payments

monthly_cum_perc_df = monthly_cum_sdf.divide(contract_values, axis=1)

monthly_cum_perc_df.plot(figsize=(20,8), legend=False, title="Monthly Cumulative % Payments for 100 Random Contracts in Jan 2019 Cohort")
ax.set_xlabel("Transaction Date")
ax.set_ylabel("Total Repayment Amount % of Contract Value")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0%}'.format(x)))
plt.savefig('Monthly Cumulative percentage Payments for 100 Random Contracts in Jan 2019 Cohort')


## HISTAGRAM OF daily % rePAYMENTS

payment_series_array =  percent_df.stack().to_numpy()
daily_nonzero_percent_payments = payment_series_array[payment_series_array.nonzero()]

f, ax = plt.subplots(figsize=(7, 5))
plt.yscale('log')
ax = sns.histplot(daily_nonzero_percent_payments, bins=100, )
ax.set_title('Daily Payment Histogram')
ax.set_xlabel('Payment %')
ax.set_ylabel('Count')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0%}'.format(x)))
plt.savefig('Daily Payment Histogram.png')



## monthly HISTAGRAM OF daily % rePAYMENTS


df = pd.merge(
    monthly_sdf.T,
    cdf,
    how='inner',
    left_index=True,
    right_index=True)

percent_month_df = df.divide(df['TotalContractValue'], axis=0).drop(columns=['TotalContractValue'])
month_payment_series_array =  percent_month_df.stack().to_numpy()
monthly_nonzero_percent_payments = month_payment_series_array[month_payment_series_array.nonzero()]

f, ax = plt.subplots(figsize=(7, 5))
#plt.yscale('log')
ax = sns.histplot(monthly_nonzero_percent_payments, bins=100, )
ax.set_title('Monthly Payment Histogram')
ax.set_xlabel('Payment %')
ax.set_ylabel('Count')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0%}'.format(x)))
plt.savefig('Monthly Payment Histogram.png')
