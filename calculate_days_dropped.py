# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:23:07 2020

@author: mark
"""

import pandas as pd
from matplotlib import pyplot as plt

from individual_analysis1 import create_small_df, PAYMENT_DATA_SQL

import google_sa_auth


"""
To dO, lets investigate the r/ship between sum(days dropped) and (total [%?] paid)
then build a classification model, then later a time series model for the worse contracts
each classification can have different prior for bayes
"""


df = pd.read_pickle('files\\small_df.pkl')

df = df.groupby(['ContractId', 'TransactionTS']).sum()

daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='D', level=1)]).sum()

daily_sdf['TransactionTS'] = daily_sdf.index.get_level_values(1)

daily_sdf[['prev_payment_date', 'prev_duration']] = daily_sdf.groupby(level=0)[['TransactionTS', 'Duration']].shift(1)

## days since token dropped
# WHAT HAPPENS TO ADJUSTMENTS
daily_sdf['days_dropped'] = (daily_sdf['TransactionTS']  - daily_sdf['prev_payment_date'] - pd.to_timedelta(daily_sdf['prev_duration'], unit='D')).dt.days




# ## smaller df - these guys are interesting because they both pay back the same but one is regular payer and the other is bulk
# small_daily_sdf = daily_sdf.loc[['1574640',  '1574676']]

# pivoted = small_daily_sdf[['days_dropped', 'AmountPaid']].unstack('ContractId').fillna(0).sort_index().cumsum(axis=0)

# pivoted.plot(secondary_y=[('days_dropped', '1574640'),
#             ('days_dropped', '1574676'),], legend=False) 



# small_df2 = create_small_df(size=1000, use_monthdiff=True, random_seed=42)

# cum_small_df = small_df2.groupby('ContractId').cumsum(axis=0)
# #TO DO - sort out percentages
# final_paid = cum_small_df.loc(axis=0)[:,22]

# fig, ax = plt.subplots()
# plt.scatter(final_paid['AmountPaid'],final_paid['Duration'])

total_paid = daily_sdf.groupby('ContractId').sum()
monthly_sdf = daily_sdf.groupby(['ContractId', pd.Grouper(freq='M', level=1)]).sum()
monthly_cumsum = monthly_sdf.groupby('ContractId').cumsum(axis=0)
six_month_mark = monthly_cumsum.loc(axis=0)[:,'2019-06-30 00:00:00+00:00'] ##be careful - we lose some contracts that havent paid in June

contract_sql = """
    SELECT c.ContractId,
            c.MainApplicantGender, 
            c.Age, 
            c.Region,
            c.Town,
            c.Occupation, 
            c.Product,
        Price + AdditionalFee as TotalContractValue,     
    FROM `afcproj.files_dupe.Contracts_20201117` c
    join `afcproj.files_dupe.jan_19_cohort` j
        on c.ContractId = j.ContractId
        """
cfdf = pd.read_gbq(contract_sql, index_col='ContractId', dialect='standard')  #.astype('float64')

all_features = pd.merge(total_paid,
         cfdf,
         how='inner',
         left_index=True,
         right_index=True).sort_index()


all_features = pd.merge(all_features,
         six_month_mark,
         how='inner',
         left_index=True,
         right_index=True,
         suffixes=['_total','_6m']).sort_index()



all_features['percent_paid'] = all_features['AmountPaid_total']/all_features['TotalContractValue']
fig, ax = plt.subplots()
plt.scatter(all_features['days_dropped_6m'], all_features['percent_paid'])


fig, ax = plt.subplots()
plt.scatter(all_features['AmountPaid_6m'], all_features['percent_paid'])
