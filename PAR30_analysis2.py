# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 10:22:52 2021

@author: mark
"""


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from basic_datasets import BasicDatasets

bd = BasicDatasets()
full_ts = bd.daily_full_ts
dts = bd.daily_ts #excludes fully paid
dfp = bd.daily_fully_paid
#plot number of par30s against final loss

full_ts['PAR30+'].diff().groupby(level=0).sum()

par30_daily_pivot = full_ts['PAR30+'].unstack(0).mask(dfp)

num_defaults = par30_daily_pivot.diff().replace(-1,0).sum() 

dcpp = bd.daily_cumulative_percent_sdf_pivot
final_value = dcpp.iloc[-1]
final_loss = 1 - final_value

plt.scatter(num_defaults, final_loss)

cum_num_defaults = dfts['PAR30+'].unstack(0).diff().replace(-1,0).cumsum()

month = '2018-09-30'
month = '2019-09-30'
month = '2020-09-30'

plt.scatter(cum_num_defaults.loc[month], rmds.loss_on_remainder.loc[month])

total_dooep = dts.elec_is_off.unstack(0).cumsum()

plt.scatter(total_dooep.loc[month], rmds.loss_on_remainder.loc[month])


