# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:54:53 2021

@author: mark
"""

import seaborn as sns
import pandas as pd

from individual_analysis1 import create_percent_sdf
from basic_datasets import BasicDatasets, MONTHLY_PERCENT_PIVOT

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from scipy.stats import kurtosis

import datetime as dt


bd = BasicDatasets()
monthly_diff = bd.monthly_percent_pivot.diff().mask(bd.monthly_fully_paid)
monthly_diff.plot(legend=False)

daily_diff = bd.daily_percent_pivot.diff().mask(bd.daily_fully_paid)



def standardise(diff_df):    
    mu = diff_df.loc['Feb-2018':].stack().mean()
    sigma = diff_df.loc['Feb-2018':].stack().std()

    stdized = (diff_df - mu) /sigma

    s = stdized.loc['Feb-2018':].stack()
    return mu, sigma, s

m_mu, m_sigma, ms = standardise(monthly_diff)
print(kurtosis(ms))

d_mu, d_sigma, ds = standardise(daily_diff)
print(kurtosis(ds))

## +- 30 SDs
sns.histplot(ms, bins=25)
plt.hist(ms, bins=250)

s.describe()
from scipy.stats import normaltest

stat, p = normaltest(s)


s.sort_values().head(10)







## ARMA MODEL

pred_start_date = dt.datetime(2020, 7, 31,)
pred_end_date = dt.datetime(2020, 11, 30,)

cid = '1349704'
one_contract_ts = monthly_diff[cid].loc['Feb-2018':]
model = ARIMA(one_contract_ts, order=(0, 0, 1))
results = model.fit(disp=True)
arma_prediction = results.predict(
    start=pred_start_date, end=pred_end_date, dynamic=False)
plt.figure(figsize=(10, 5))
l1, = plt.plot(one_contract_ts.cumsum(), label='Observation')
l2, = plt.plot(arma_prediction.cumsum(), label='ARMA')
plt.legend(handles=[l1, l2])
plt.savefig('ARMA prediction {}'.format(contractid), bbox_inches='tight', transparent=False)
plt.close()
print('ARMA MAE = ', mean_absolute_error(arma_prediction, test))

model = VAR(monthly_diff.fillna(0))
results = model.fit(12)
results.summary()