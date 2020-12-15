# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:10:37 2020

@author: mark
"""

import seaborn as sns

from matplotlib import pyplot as plt

from individual_analysis1 import create_small_df, create_percent_sdf, convert_to_daily_pivot

small_df = create_small_df(size=1000, cohort='dec_17')
daily_sdf_pivot = convert_to_daily_pivot(small_df)     
cumulative_percent_sdf = create_percent_sdf(daily_sdf_pivot, cumulative=True, cohort='dec_17')

at_month_18 = cumulative_percent_sdf.loc['2019-05-31']
last_chance = cumulative_percent_sdf.loc['2020-11-17']

paid_off_on_time = (at_month_18 >= 0.98)
paid_off_eventually = (at_month_18 < 0.98) & (last_chance >= 0.98)
not_paid_ok = (last_chance > 0.8) & (last_chance < 0.980)
bad = (last_chance < 0.8) & (last_chance >= 0.5)
very_bad = (last_chance < 0.5) & (last_chance >= 0.2) 
default = (last_chance < 0.2)  

## This guy is interesting:
# cumulative_percent_sdf['1351574'].plot()


categories = {'Paid off on time':paid_off_on_time,
              'Paid off after 18 months':paid_off_eventually ,
              'Not fully repaid':not_paid_ok ,
              'Less than 80% paid':bad ,
              'Less than 50% paid':very_bad ,
              'Less than 20% paid':default,

              }

run_sum = 0
for cat_name, cat in categories.items():
    print(cat_name, cat.sum())
    run_sum += cat.sum()
## should add to 1000!
print(run_sum)


cumulative_percent_sdf[paid_off_on_time.index[paid_off_on_time]].plot()


percent_sdf = create_percent_sdf(daily_sdf_pivot, cumulative=False, cohort='dec_17')



## look at individual and average variance
mean_daily_repayment = {}
std_daily_repayment = {}
std_cust_repayment = {}
full_cumulative_repayment = {}



for cat_name, cat in categories.items():
    mean_daily_repayment[cat_name] = percent_sdf[cat.index[cat]].mean(axis=1)
    std_daily_repayment[cat_name] = percent_sdf[cat.index[cat]].std(axis=1)
    std_cust_repayment[cat_name] = percent_sdf[cat.index[cat]].std(axis=0)
    full_cumulative_repayment[cat_name] = cumulative_percent_sdf[cat.index[cat]]
    
    # daily average payments across the group
    fig, ax = plt.subplots()
    mean_daily_repayment[cat_name].plot(title=cat_name, legend=False)
    plt.savefig('files\\{}_daily_average_payments'.format(cat_name))
    
    ## what do the cumulative payhments look like?
    fig, ax = plt.subplots()
    full_cumulative_repayment[cat_name].plot(title=cat_name, legend=False)
    plt.savefig('files\\{}_cumulative_payments'.format(cat_name))
    
    ##histograms of daily payments
    fig, ax = plt.subplots()
    sns.histplot(percent_sdf[cat.index[cat]], bins=100, legend=False)
    plt.xscale('log')
    plt.title('Histogram of Daily Payments {}'.format(cat_name))
    plt.savefig('Histogram of Daily Payments {}')
    
    ##what are the total number of token bought for each group?
    
plt.show()