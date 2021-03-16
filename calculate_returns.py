# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:38:04 2021

@author: mark
"""

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats


from basic_datasets import BasicDatasets

bd = BasicDatasets()

"""
df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
df = df.groupby(['ContractId', 'TransactionTS']).sum()
daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
daily_sdf_pivot = daily_sdf['AmountPaid'].unstack(0).fillna(0).sort_index()
"""

dcpp = bd.daily_cumulative_percent_sdf_pivot
mcpp = bd.monthly_cumulative_percent_sdf_pivot
fully_paid = bd.monthly_fully_paid
cdf = bd.contract_values

monthly_durations = bd.df.groupby(['ContractId', pd.Grouper(freq='M', level=1)])['Duration'].sum()

fts = bd.daily_full_ts
total_tokens_bought = fts['Duration'].groupby('ContractId').sum()
cumulative_tokens = monthly_durations.groupby('ContractId').cumsum()
fully_paid_tokens = total_tokens_bought[fully_paid.iloc[-1]]

##this is interesting
fully_paid_tokens.plot()


assumed_tokens_to_repay = fully_paid_tokens.mode().values[0]
margin = 0.0

final_value = dcpp.iloc[-1]
final_loss = 1 - final_value


""" 
example: EL at t0 = 80% 
loss at the end = 10%
after 30% paid: 70% left to pay, 10% goes unpaid: 10/70 = 14%
"""

remainder = 1 - mcpp 
loss_on_remainder = (final_loss/remainder).mask(fully_paid)

month = '2020-03-31'
ctp = cumulative_tokens.unstack(0).ffill(axis=0)
pd.concat([ctp.loc[month], final_loss], axis=1)

for month in ['2018-03-31', '2019-03-31', '2020-03-31']:
    fig, ax = plt.subplots()
    #ax.scatter(x=ctp.loc[month], y=final_loss)
    ax.scatter(x=ctp.loc[month], y=loss_on_remainder.loc[month])
    plt.title(month)
    fig, ax = plt.subplots()
    #ax.scatter(x=mcpp.loc[month], y=final_loss)
    ax.scatter(x=mcpp.loc[month], y=loss_on_remainder.loc[month]) ## looks best
    ax.set_ylim(0,1)
    plt.title(month)

def EL_model(month):
    model = LinearRegression()
    x=mcpp.loc[month].values.clip(0,1)
    y=loss_on_remainder.loc[month].values.clip(0,1)
    
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
    
    model.fit(x[mask].reshape((-1, 1)), y[mask])
    model.predict(np.array([0.5, 0.6, 0.7]).reshape((-1, 1)))
    
def portfolio_expected_loss(dcpp):
    final_value = dcpp.iloc[-1]
    final_loss = 1 - final_value
    PEL = final_loss.mean()
    final_loss.plot(kind='hist')
    return PEL
    
PEL = portfolio_expected_loss(mcpp)

one_contract = mcpp.columns[3]
one_ts = mcpp[one_contract]

## Assume cost and v0 are the same for now (aka margin=0)
v_0 = 1 - PEL - margin
expected_monthly_repayment = 1/len(one_ts.index)
df = one_ts.to_frame('paid')
df['emr'] = expected_monthly_repayment
df['cum_emr'] = df['emr'].cumsum()
v = {}
for i, cum_payment in enumerate(one_ts):
    v[i+1] = cum_payment + (1-cum_payment)*(1-PEL) - expected_monthly_repayment
    
one_ts_vals = one_ts + (1-one_ts)*(1-PEL) - df['cum_emr']

vals = (mcpp + (1-mcpp)*(1-PEL)).subtract(df['cum_emr'], axis=0).mask(fully_paid)
returns = vals.diff()

returns.corr().stack().describe()

""" 
    both 50% paid in month 2 and 50% in month 3 will have high values (high returns) 
    so will be non-zero correlated.
"""

"""
Can we bck out Rsq from par30 flags?
"""