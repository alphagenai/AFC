# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:47:42 2021

@author: mark
"""


from afc_cohort_timeseries import build_cohort_repayment_schedule, prettify_dfs_for_output
    
    

"""
cohort forecast
"""


perc_orig_balance_timeseries_df, amort_timeseries_df, cohort_contract_sum_df = build_cohort_repayment_schedule()


amort_df_to_plot = prettify_dfs_for_output(amort_timeseries_df).loc[:,0:42]
df = perc_df_to_plot = prettify_dfs_for_output(perc_orig_balance_timeseries_df).loc[:,0:42]


ax = perc_df_to_plot.loc[:, 18:].T.plot(xticks=range(42), figsize=(20,8), legend=False, title="Amortization as % of Original Balance")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0%}'.format(x) ))
 

""" Curve fitting """

## Might want to remove November 2020 as not complete
df = df.mask(df.eq(df.min(axis=1), axis=0),)

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd


def f1(x, a, b, c):
    return a*x**2 + b*x + c

def f2(x, a, b, c, d, e):
    return d / (b + c*np.exp(e+a*x))

# def f3(x, a, b):
#     return -a * x**(-b)

def f3(x, a, b, c, d):
    return a + (d/ (1+np.exp(b*(x-c))))

def f4(x, a, b, c, d):
    return a + (b-a)*(1-np.exp(-np.exp(c*(np.log(x)-np.log(d)))))


x_fcast = df.columns
y = df.loc['Jun-17':'Dec-17', 18:34]
x = y.index
ax = y.T.plot()


popt, pcov = curve_fit(f1, x, y,)
ax.plot(x_fcast, f1(x_fcast, *popt), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

popt, pcov = curve_fit(f2, x, y,)
ax.plot(x_fcast, f2(x_fcast, *popt), 'r--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt[0:3]))


popt, pcov = curve_fit(f3, x, y,)
ax.plot(x_fcast, f3(x_fcast, *popt), 'b--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt[0:3]))


#popt, pcov = curve_fit(f4, x, y,)
#ax.plot(x_fcast, f4(x_fcast, *popt), 'p--',
#         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt[0:3]))
