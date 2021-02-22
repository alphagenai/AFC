# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:10:27 2020

@author: mark
"""

import pandas as pd
import matplotlib.ticker as ticker

from matplotlib import pyplot as plt

from individual_analysis1 import create_small_df, create_percent_sdf, convert_to_daily_pivot
from basic_datasets import MONTHLY_PERCENT_PIVOT, MONTHLY_CUMULATIVE_PERCENT_SDF_PIVOT, MONTHLY_FULLY_PAID
from autocor_analysis import best_params

""" There is evidence that cohort-level curves are solely due to completed contracts 
TO DO: EVALUATE THIS MODEL AGAINST ACTUAL 6 MONTH REALISATIONS
"""

def six_month_ma(counterparty, forecast_startdate,  ma_pivot_incl0):
        return ma_pivot_incl0.loc[forecast_startdate, counterparty.ContractId]

def f(w):                        
    def g(x):
        return (w*x).sum()
    return g

if __name__ == "__main__":        
    
    mpp = MONTHLY_PERCENT_PIVOT
    mcp = MONTHLY_CUMULATIVE_PERCENT_SDF_PIVOT
    mfp = MONTHLY_FULLY_PAID
    ma_pivot_incl0 = mpp.iloc[1:].rolling(window=6).mean().shift(1)

    weighted_ma = mpp.iloc[1:].rolling(window=6).apply(f(best_params))

    ## 6 MONTH FORECAST FROM THE 18TH MONTH
    forecast = pd.concat([mpp[0:18] ,ma_pivot_incl0.iloc[18:24]], axis=0).cumsum(axis=0).clip(upper=1.)

    remaining = 1 - mcp
    expected_remaining_time = (remaining / ma_pivot_incl0).mask(mfp).clip(lower=0,upper=60)
    td = (30*expected_remaining_time).apply(lambda x:pd.to_timedelta(x, unit='D'))
    expected_repayment_date = td.add(mcp.index.values, axis=0)

def plot():    
    ax = mpp.cumsum(axis=0)[mpp.columns[0:10]].plot(legend=False)
    forecast[mpp.columns[0:10]][-6:].plot(ax=ax, legend=False, color='red', style=['--' for col in forecast.columns])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0%}'.format(x)))
    title = 'Moving Average 6 Month Forecast'
    plt.title(title)
    plt.savefig('files\\'+title)

