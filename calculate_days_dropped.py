# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:23:07 2020

@author: mark
"""

from matplotlib import pyplot as plt

from individual_analysis1 import create_small_df

# daily_sdf = small_df.groupby(['ContractId', pd.Grouper(freq='D', level=1)]).sum()

# daily_sdf['TransactionTS'] = daily_sdf.index.get_level_values(1)

# ## unstack here?
# #daily_sdf['prev_payment_date'] = daily_sdf['TransactionTS'].shift(1)

# daily_sdf[['prev_payment_date', 'prev_duration']] = daily_sdf.groupby(level=0)[['TransactionTS', 'Duration']].shift(1)

# ## days since token dropped
# # WHAT HAPPENS TO ADJUSTMENTS
# daily_sdf['days_dropped'] = (daily_sdf['TransactionTS']  - daily_sdf['prev_payment_date'] - pd.to_timedelta(daily_sdf['prev_duration'], unit='D')).dt.days

# ## smaller df - these guys are interesting because they both pay back the same but one is regular payer and the other is bulk
# small_daily_sdf = daily_sdf.loc[['1574640',  '1574676']]

# pivoted = small_daily_sdf[['days_dropped', 'AmountPaid']].unstack('ContractId').fillna(0).sort_index().cumsum(axis=0)

# pivoted.plot(secondary_y=[('days_dropped', '1574640'),
#             ('days_dropped', '1574676'),], legend=False) 



small_df2 = create_small_df(size=1000, use_monthdiff=True, random_seed=42)

cum_small_df = small_df2.groupby('ContractId').cumsum(axis=0)
#TO DO - sort out percentages
final_paid = cum_small_df.loc(axis=0)[:,22]

fig, ax = plt.subplots()
plt.scatter(final_paid['AmountPaid'],final_paid['Duration'])

