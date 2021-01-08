# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:02:22 2021

@author: mark
"""


import seaborn as sns

monthly_sdf_pivot.corr()

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