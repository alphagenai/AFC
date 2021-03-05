# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 21:34:24 2021

@author: mark
"""


from sklearn import linear_model

from basic_datasets import BasicDatasets
from sklearn.model_selection import TimeSeriesSplit

bd = BasicDatasets()

mfp = bd.monthly_fully_paid

dfts = bd.daily_ts #full dataset excluding paid off contracts
monthly_days_of_elec = (~dfts['elec_is_off']).groupby(['ContractId', pd.Grouper(freq='1M', level=1)]).sum()
mde_pivot = monthly_days_of_elec.unstack(0)
mde_perc_pivot = mde_pivot.divide(monthly_elec_pivot.index.daysinmonth, axis=0)


reg = linear_model.LinearRegression()

ds = mde_perc_pivot.loc['Jan-2018':'Jul-2018'].dropna(axis=1, how='any')
X_train = ds.loc['Jan-2018':'Jun-2018'].T
y_train = ds.loc['July-2018'].T

reg.fit(X_train,y_train)
reg.coef_
y_pred = reg.predict(X_test)

tscv = TimeSeriesSplit(6, max_train_size=6)


for train_index, test_index in tscv.split(mde_perc_pivot):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

