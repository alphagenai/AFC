# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:56:29 2020

@author: mark
"""



import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from individual_analysis1 import create_small_df, create_percent_sdf, convert_to_daily_pivot

small_df = create_small_df(size=1000, cohort='dec_17')
daily_sdf_pivot = convert_to_daily_pivot(small_df)
daily_percent_pivot = create_percent_sdf(daily_sdf_pivot, cumulative=False, use_monthdiff=False, cohort='dec_17')


# Monthly analysis

#daily_sdf_pivot.index = pd.to_datetime(daily_sdf_pivot.index,format='%Y/%m/%d %H:%M:%S')
monthly_percent_pivot = daily_percent_pivot.groupby(pd.Grouper(freq='M')).sum()

d = pd.date_range(start='2017-12-01', end='2020-11-30', freq='M')

for month in d[18:20]:
    fig,ax = plt.subplots()
    sns.histplot(monthly_sdf_pivot.loc[d], bins=100, legend=False)
    title = 'Histogram for payments in {} {}'.format(month.month, month.year)
    plt.title(title)
    plt.savefig('files\\{}'.format(title))
    plt.close()
    
