# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 21:34:24 2021

@author: mark
"""

import pandas as pd
import statsmodels.api as sm

from sklearn import linear_model
from matplotlib import pyplot as plt

from basic_datasets import BasicDatasets
from sklearn.model_selection import TimeSeriesSplit



""" OKAY BUT BEAR IN MIND _OF COURSE_ theres going to be high predictability of electricity
    usage because any unused credits from larger purchases get carried over to the next month 
    Need to think about the kurtosis of payment distribution AND the relationship between 
    historic payment amounts, amount left to pay and future electricity usage """

bd = BasicDatasets()

mfp = bd.monthly_fully_paid

dfts = bd.daily_ts #full dataset excluding paid off contracts
monthly_days_of_elec = (~dfts['elec_is_off']).groupby(['ContractId', pd.Grouper(freq='1M', level=1)]).sum()
mde_pivot = monthly_days_of_elec.unstack(0).mask(mfp)
mde_perc_pivot = mde_pivot.divide(mde_pivot.index.daysinmonth, axis=0)


reg = linear_model.LinearRegression()

ds = mde_perc_pivot.loc['Jan-2018':'Jul-2018'].dropna(axis=1, how='any')
X_train = ds.loc['Jan-2018':'Jun-2018'].T
y_train = ds.loc['July-2018'].T

reg.fit(X_train,y_train)
reg.coef_
y_pred = reg.predict(X_test)

# tscv = TimeSeriesSplit(6, max_train_size=6)


# for train_index, test_index in tscv.split(mde_perc_pivot):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

ds_list = []
X_train_list = []
X_test_list = {}

for i in range(len(mde_perc_pivot)-8+1): #+1 for start month 
    ## start in Jan (index 1) end at 6months_x + 1 month_y + 1 bcos python 
    ds = mde_perc_pivot.iloc[i+1:i+8].dropna(axis=1, how='any')
    X_train = ds.iloc[0:6].T
    y_train = ds.iloc[6].T
    reg.fit(X_train,y_train)
    #print(y_train.name, reg.coef_)
    ds_list.append(ds.reset_index().drop(columns=['TransactionTS']))


full_train_set = pd.concat(ds_list, axis=1).T

X_train, y_train = full_train_set.loc[:, 0:5], full_train_set[6]
reg.fit(X_train,y_train)
print(y_train.name, reg.coef_)

mod = sm.OLS(y_train, sm.add_constant(X_train))
res = mod.fit()

print(res.summary())

fig, ax = plt.subplots()
x = X_train.iloc[:, -1]
ax.scatter(X_train.iloc[:, -2], y_train, color='green')
ax.scatter(x, y_train, color='red')
