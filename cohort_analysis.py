# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:36:25 2020

@author: mark
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

sns.set_theme(style="dark", palette='dark')

from afc_cohort_timeseries import make_cohort_columns_as_index, read_payment_data_from_bigquery, build_cohort_repayment_schedule, prettify_dfs_for_output, build_timeseries_of_repayments

SQL = """
    SELECT sum(p.AmountPaid) as total_amount_repaid,
        count(DISTINCT c.ContractId) as number_of_dist_contracts,
        EXTRACT(YEAR FROM RegistrationDate) AS cohort_year,
        EXTRACT(MONTH FROM RegistrationDate) AS cohort_month,
    FROM `afcproj.files_dupe.Payments_2020_11_17` p
    join `afcproj.files_dupe.Contracts_20201117`  c
        on p.ContractId = c.ContractId
    WHERE p.paymentStatusTypeEntity != 'REFUSED'
        and
        p.PaymentResultTypeEntity != 'PAYMENT_FREE_CREDIT'
        and c.PaymentMethod = 'FINANCED'

    group by cohort_month, cohort_year

    """
cohort_total_repaid = pd.read_gbq(SQL,)
    
SQL = """
     SELECT count(c.ContractId) as number_of_contracts,
         --count(DISTINCT c.ContractId) as dist_contracts
        SUM(Price) + SUM(AdditionalFee) as TotalContractValue, 
        
        EXTRACT(YEAR FROM RegistrationDate) AS cohort_year,
        EXTRACT(MONTH FROM RegistrationDate) AS cohort_month,
    FROM `afcproj.files_dupe.Contracts_20201117`  c
    WHERE c.PaymentMethod = 'FINANCED'
    group by cohort_month, cohort_year
    """
    
cohort_contract_value = pd.read_gbq(SQL,)
cohort_contract_value = make_cohort_columns_as_index(cohort_contract_value)
cohort_contract_value.to_csv('Cohort Contract Balances.csv')

df = pd.merge(
    cohort_total_repaid,
    cohort_contract_value,
    how='outer',
    left_on=['cohort_year', 'cohort_month'],
    right_on=['cohort_year', 'cohort_month'],
    )

df = make_cohort_columns_as_index(df)


df['total_amount_repaid'] = df['total_amount_repaid'].astype('float64')
df['mean_value_per_contract'] = df['TotalContractValue']/df['number_of_contracts']
df['mean_repay_per_contract'] = df['total_amount_repaid']/df['number_of_contracts']
df['Percent Repaid'] = df['total_amount_repaid']/df['TotalContractValue']
#df['Percent Repaid'] = df['TotalContractValue']/df['total_amount_repaid']  #what does this value represent?
df['Total Amount Repaid'] = df['total_amount_repaid']

perc_orig_balance_timeseries_df, amort_timeseries_df, cohort_contract_sum_df = build_cohort_repayment_schedule()



average_perc_cum_repayment_by_month = 1 - perc_orig_balance_timeseries_df.mean(axis=0)
average_perc_repayment_per_month_by_cohort = perc_orig_balance_timeseries_df.diff(axis=1).mean(axis=1)
average_perc_repayment_by_month = perc_orig_balance_timeseries_df.diff(axis=1).mean(axis=0)


timeseries_df = build_timeseries_of_repayments()
mean_perc_repayment_by_cohort = timeseries_df.mean(axis=1).divide(cohort_contract_value['TotalContractValue'] )


#### Create Charts for Presentation
##  Cohort Size
ccv = prettify_dfs_for_output(cohort_contract_value, end_date='2020-10-30')
ax = ccv[['number_of_contracts', 'TotalContractValue']].plot(kind='bar', figsize=(20,8), secondary_y=['TotalContractValue'])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
ax.right_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000000)+'m'))
ax.set_xlabel("Cohort Origination Month")

plt.savefig('Number of contracts+contract value.png')

## Total Repayment
ax = prettify_dfs_for_output(df[['Total Amount Repaid', 'TotalContractValue', 'Percent Repaid']], end_date='2020-10-30').plot(kind='bar', secondary_y=['Percent Repaid'], figsize=(20,8))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000000)+'m'))
ax.right_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0%}'.format(x)))
ax.set_xlabel("Cohort Origination Month")
ax.set_ylabel("Total Repayment")
plt.savefig('Cohort Total Repayment.png')


## Average Repayment over time
ax = timeseries_df.sum(axis=0).loc[0:40].plot(kind='bar', figsize=(20,8))
ax.set_xlabel("Months Since Origination")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000000)+'m'))
plt.savefig('Average Repayment Through Time.png')

######### Want to create an average repayment as % of original contract balance so far...
#perc_orig_balance_timeseries_df.sum(axis=0).plot(kind='bar', figsize=(20,8))  #cant sum percentages
#perc_orig_balance_timeseries_df.mean(axis=0)  #this is the mean using no of cohorts, not number of contracts


## Average Monthly Repayment by Cohort
mean_repayments_df = prettify_dfs_for_output(
    pd.merge(
        left=timeseries_df.mean(axis=1).rename('Mean Repayment'),
        right=mean_perc_repayment_by_cohort.rename('Mean % Repayment'),
        left_index=True,
        right_index=True,
        ), 
    end_date='2020-10-30',
    )
ax = mean_repayments_df.plot(kind='bar', secondary_y=['Mean Repayment'],figsize=(20,8))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0%}'.format(x)))
ax.right_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000000)+'m'))
ax.set_xlabel("Cohort Origination Month")
ax.set_ylabel("Percentage Monthly Repayment")

plt.savefig('Average Cohort Repayment.png')


