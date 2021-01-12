# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:02:22 2021

@author: mark
"""


import seaborn as sns

mycorr = monthly_sdf_pivot.corr()
sns.heatmap(mycorr)
mycorr > 0.8
(mycorr > 0.8).sum(axis=1)
high_corr = (mycorr > 0.8).sum(axis=1)
high_corr.sort_values()
highest_corr = high_corr.sort_values().tail(20)
monthly_sdf_pivot.loc[:,highest_corr.index]
monthly_sdf_pivot.loc[:,highest_corr.index].to_csv('high_correlation.csv')
high_corr_ts = monthly_sdf_pivot.loc[:,highest_corr.index]
high_corr_ts.cumsum(axis=0).plot()


### I THINK WHAT WE REALLY WANT IS STATE (aka default) CORRELATION

no_NAs = pd_calc._defaults.fillna(False)
mycorr = no_NAs.corr()
no_NAs.loc[:,mycorr.isna().all()]  # NAs are all false


## par30 correlation
par30_daily_pivot = daily_sdf_fullts['PAR30+'].unstack(0).sort_index()
mycorr= par30_daily_pivot.corr()
sns.heatmap(mycorr)

par30_daily_pivot.cov()
daily_sdf_pivot.rolling(360).corr().abs().mean()
