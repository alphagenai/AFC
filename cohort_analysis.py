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

from afc_cohort_timeseries import make_cohort_columns_as_index, read_payment_data_from_bigquery, build_cohort_repayment_schedule, prettify_dfs_for_output

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

df['mean_value_per_contract'] = df['TotalContractValue']/df['number_of_contracts']
df['mean_repay_per_contract'] = df['total_amount_repaid']/df['number_of_contracts']
df['repay_per_contract_dollar'] = df['total_amount_repaid']/df['TotalContractValue']

perc_orig_balance_timeseries_df, amort_timeseries_df, cohort_contract_sum_df = build_cohort_repayment_schedule()


average_perc_cum_repayment_by_month = 1 - perc_orig_balance_timeseries_df.mean(axis=0)
average_repayment_per_month_by_cohort = perc_orig_balance_timeseries_df.diff(axis=1).mean(axis=1)
average_perc_repayment_by_month = perc_orig_balance_timeseries_df.diff(axis=1).mean(axis=0)

payadj_df_grp = read_payment_data_from_bigquery()
timeseries_df = payadj_df_grp.unstack('monthdiff').astype('float64')['sum_amount_paid']


### Create Charts for Presentation
ccv = prettify_dfs_for_output(cohort_contract_value, end_date='2020-10-30')
ax = ccv[['number_of_contracts', 'TotalContractValue']].plot(kind='bar', figsize=(20,8), secondary_y=['TotalContractValue'])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
ax.right_ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000000)+'m'))
plt.savefig('Number of contracts+contract value.png')
