# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:10:37 2020

@author: mark
"""


from individual_analysis1 import create_small_df

small_df = create_small_df(size=1000, cohort='dec_17')

at_month_18 = cumulative_percent_sdf.loc['2019-05-31']
last_chance = cumulative_percent_sdf.loc['2020-11-17']

paid_off_on_time = (at_month_18 >= 1.0)
paid_off_eventually = (at_month_18 < 1.0) & (last_chance >= 1.0)
not_paid_ok = (last_chance > 0.8) & (last_chance < 1.0)
bad = (last_chance < 0.8) & (last_chance >= 0.5)
very_bad = (last_chance < 0.5) 

## This guy is interesting:
# cumulative_percent_sdf['1351574'].plot()

num_paid_off = paid_off.sum()
num_not_good = NG.sum()
num_bad = bad.sum()

categories = [paid_off_on_time,
paid_off_eventually ,
not_paid_ok ,
bad ,
very_bad ,
]

run_sum = 0
for cat in categories:
    print(cat.sum())
    run_sum += cat.sum()
## should add to 1000!
print(run_sum)


cumulative_percent_sdf[paid_off_on_time.index[paid_off_on_time]].plot()