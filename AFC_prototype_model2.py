# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:42:18 2020

@author: mark
"""

#import pmdarima as pm
#from statsmodels.tsa.seasonal import seasonal_decompose
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
from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
#from statsmodels.tsa.statespace.sarimax import SARIMAX
#from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
#from fbprophet import Prophet
#from fbprophet.plot import plot_plotly, add_changepoints_to_plot
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam

from individual_analysis1 import create_small_df



"""
TO DO: ## whats the autocorrelation across the whole portfolio?
"""



required=1

if required:
    small_df = create_small_df(size=10, cohort='dec_17')
    sdf = small_df['AmountPaid'].unstack(0).fillna(0).sort_index()
    monthly_sdf = sdf.groupby(pd.Grouper(freq='M')).sum()
    monthly_sdf.index = monthly_sdf.index.tz_localize(None)


pred_start_date = dt.datetime(2020, 7, 31,)
pred_end_date = dt.datetime(2020, 11, 30,)


for contractid in monthly_sdf.columns:
    one_contract = monthly_sdf[contractid]
    # one_contract.to_pickle('files\\pickle_one_contract')
    # one_contract = pd.read_pickle('files\\pickle_one_contract')
    
    train = one_contract.loc[one_contract.index < pd.to_datetime(pred_start_date)]
    test = one_contract.loc[one_contract.index >= pd.to_datetime(pred_start_date)]
    train_and_eval_arima(train, test)
    
    
def train_and_eval_arima(train, test):    
    model = ARIMA(train, order=(0, 0, 1))
    try:
        results = model.fit(disp=True)
    except ValueError:
        return
    arma_prediction = results.predict(
        start=pred_start_date, end=pred_end_date, dynamic=False)
    plt.figure(figsize=(10, 5))
    l1, = plt.plot(one_contract, label='Observation')
    l2, = plt.plot(arma_prediction, label='ARMA')
    plt.legend(handles=[l1, l2])
    plt.savefig('ARMA prediction {}'.format(contractid), bbox_inches='tight', transparent=False)
    plt.close()
    print('ARMA MAE = ', mean_absolute_error(arma_prediction, test))


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


# train = train.to_frame().rename(index={'TransactionTS': 'ds'}).rename(columns={'1564609':'y'})
# train['ds'] = train.index.values
# m = Prophet()
# m.fit(train)


### XGBoost




def featurize(t):
    X = pd.DataFrame()

    X['day'] = t.index.day
    X['month'] = t.index.month
    X['quarter'] = t.index.quarter
    X['dayofweek'] = t.index.dayofweek
    X['dayofyear'] = t.index.dayofyear
    X['weekofyear'] = t.index.weekofyear
    y = t.y
    return X, y


for contractid in monthly_sdf.columns:
    one_contract = monthly_sdf[contractid]

    one_contract = one_contract.to_frame()
    one_contract.index.rename('ds', inplace=True)
    one_contract.rename(columns={one_contract.columns[0]:'y'}, inplace=True)


    dataset = one_contract
    featurize(dataset)[0].head()
    
    X_train, y_train = featurize(
        dataset.loc[dataset.index < pd.to_datetime(pred_start_date)])
    X_test, y_test = featurize(
        dataset.loc[dataset.index >= pd.to_datetime(pred_start_date)])
    
    scaler = StandardScaler()
    scaler.fit(X_train)
        
    scaled_train = scaler.transform(X_train)
    scaled_test = scaler.transform(X_test)
    
    XGBOOST_model = XGBRegressor(n_estimators=7)
    
    XGBOOST_model.fit(scaled_train, y_train,)
    #                  eval_set=[(scaled_train, y_train), (scaled_test, y_test)],
    #                  verbose=True)
    XGBOOST_prediction = XGBOOST_model.predict(scaled_test)
    
    print('XGBOOST MAE = ', mean_absolute_error(XGBOOST_prediction, y_test))
    
    
    XGBOOST_df = pd.DataFrame({'y': XGBOOST_prediction.tolist()})
    XGBOOST_df.index = y_test.index
    
    plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots()
    ax.plot(dataset.tail(50))
    ax.plot(XGBOOST_df.tail(50))
    ax.set_title("XGBOOST")
    
    for ax in fig.get_axes():
        ax.label_outer()
    fig.autofmt_xdate()
