# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:48:47 2020

@author: mark
"""

import pandas as pd
import datetime as dt
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder

from matplotlib import pyplot as plt

from individual_analysis1 import create_small_df, create_cumulative_percent_sdf, convert_to_daily



## assemble features

if 0:
    contract_sql = """
        SELECT c.*,
            Price + AdditionalFee as TotalContractValue,     
        FROM `afcproj.files_dupe.Contracts_20201117` c
        join `afcproj.files_dupe.jan_19_cohort` j
            on c.ContractId = j.ContractId
            """
    cdf = pd.read_gbq(contract_sql,index_col='ContractId') #.astype('float64')
    
    
    all_features = pd.merge(small_df,
             cdf,
             how='inner',
             left_index=True,
             right_index=True).sort_index()


### TO DO: 
""" 
    1. Try some feature engineering like [time since last payment] and [size of last payment]
    2. Treat every day like a random variable with baysian probability of receiving a payment
"""

#### Super Simple Regression Model
#currently only works for one contract at a time

if required:
    small_df = create_small_df()
    sdf = small_df['AmountPaid'].unstack(0).fillna(0).sort_index()
    monthly_sdf = sdf.groupby(pd.Grouper(freq='M')).sum()
    
    df = create_cumulative_percent_sdf(monthly_sdf)
    
    contractIds = df.columns[0:3]
    df = df[contractIds] 
    #diff_df = df.diff()
    
    monthly = df.stack().reset_index()
    
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
              SVR(),
              svr_rbf,
              svr_lin,
              svr_poly,
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
    
    
