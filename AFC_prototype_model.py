# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:48:47 2020

@author: mark
"""

import pandas as pd
import datetime as dt
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, TheilSenRegressor, TweedieRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder

from matplotlib import pyplot as plt

from individual_analysis1 import create_small_df, create_cumulative_percent_sdf, convert_to_daily





### TO DO: 
""" 
    1. Try some feature engineering like [time since last payment] and [size of last payment]
    2. Treat every day like a random variable with baysian probability of receiving a payment
    3. Try clustering to see if an algo can identify regular/stepper payers
"""

#### Super Simple Regression Model
#currently only works for one contract at a time

if required:
    small_df = create_small_df()
    sdf = small_df['AmountPaid'].unstack(0).fillna(0).sort_index()
    monthly_sdf = sdf.groupby(pd.Grouper(freq='M')).sum()
    
    monthly_cumulative_percent_sdf = create_cumulative_percent_sdf(monthly_sdf)
    
    contractIds = df.columns[0:3]
    df = df[contractIds] 
    #diff_df = df.diff()
    
    monthly = monthly_cumulative_percent_sdf.stack().reset_index()
    
    monthly.to_pickle('monthly')
monthly = pd.read_pickle('monthly')

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
one_hot_contractIds = enc.fit_transform(monthly['ContractId'].to_numpy().reshape(-1,1))
ordinal_dates = monthly['level_0'].map(dt.datetime.toordinal).to_numpy().reshape(-1,1)
X_train = np.concatenate((ordinal_dates, one_hot_contractIds), axis=1)  #cant concatenate dense and sparse
y = monthly[0].to_numpy()
#model = LinearRegression()
#model = RandomForestRegressor()


svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

for model in [
              #LinearRegression(),
              #RandomForestRegressor(),
              #Ridge(),
              #HistGradientBoostingRegressor(),
              #LogisticRegression(),
#              SVR(),
#              svr_rbf,
#              svr_lin,
#              svr_poly,
#              TheilSenRegressor(),
              TweedieRegressor(power=2, link='log')
              ]:
    fit_and_plot(model)
    
def fit_and_plot(model):
    model.fit(X_train, y)
    
    future_X = np.concatenate([np.ones([3, 1])*dt.datetime(2021,11,1).toordinal(), np.eye(3)], axis=1)
    X_test = np.concatenate([X_train,future_X], axis=0)
    y_pred = model.predict(X_test)
    
    
    
    # Plot
    fig,ax = plt.subplots()
    for i in range(one_hot_contractIds.shape[1]):
        plt.scatter(X_train[:,0][X_train[:,i+1].nonzero()], y[X_train[:,i+1].nonzero()],  color='black')
        plt.plot(X_test[:,0][X_test[:,i+1].nonzero()], y_pred[X_test[:,i+1].nonzero()], color='blue', linewidth=3)
        plt.title('{}'.format(model))
    plt.show()
    
    
