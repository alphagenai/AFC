# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:42:18 2020

@author: mark
"""

#import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import datetime
import requests
import warnings

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import datetime as dt

#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import StandardScaler
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam

from individual_analysis1 import create_small_df

required=0

if required:
    small_df = create_small_df(size=100)
    sdf = small_df['AmountPaid'].unstack(0).fillna(0).sort_index()
    monthly_sdf = sdf.groupby(pd.Grouper(freq='M')).sum()
    
    one_contract = monthly_sdf[monthly_sdf.columns[0]]
    one_contract.index = one_contract.index.tz_localize(None)
    one_contract.to_pickle('pickle_one_contract')

one_contract = pd.read_pickle('pickle_one_contract')

start_date = dt.datetime(2020, 7, 31,) #tzinfo=dt.timezone.utc)
end_date = dt.datetime(2021, 7, 31,) #tzinfo=dt.timezone.utc)
end_date = dt.datetime(2020, 11, 30,) #tzinfo=dt.timezone.utc)


train = one_contract.loc[one_contract.index < pd.to_datetime(start_date)]
test = one_contract.loc[one_contract.index >= pd.to_datetime(start_date)]


### SARIMAX

model = SARIMAX(train, order=(2, 1, 3))
results = model.fit(disp=True)
sarimax_prediction = results.predict(
    start=start_date, end=end_date, dynamic=False)
plt.figure(figsize=(10, 5))
l1, = plt.plot(one_contract, label='Observation')
l2, = plt.plot(sarimax_prediction, label='ARIMA')
plt.legend(handles=[l1, l2])
plt.savefig('SARIMAX prediction', bbox_inches='tight', transparent=False)

print('SARIMAX MAE = ', mean_absolute_error(sarimax_prediction, test))


model = pm.auto_arima(train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3,  # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

print(model.summary())



### Prophet


train = train.to_frame().rename(index={'TransactionTS': 'ds'}).rename(columns={'1564609':'y'})
train['ds'] = train.index.values
m = Prophet()
m.fit(train)