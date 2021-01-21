# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:10:27 2020

@author: mark
"""

import pandas as pd
import matplotlib.ticker as ticker

from matplotlib import pyplot as plt

from individual_analysis1 import create_small_df, create_percent_sdf, convert_to_daily_pivot


""" There is evidence that cohort-level curves are solely due to completed contracts """

if __name__ == "__main__":        
    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    df = df.groupby(['ContractId', 'TransactionTS']).sum()

    monthly_sdf = df.groupby(['ContractId', pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
    monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)

    monthly_percent_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=False, cohort='dec_17')
    
    ma = monthly_percent_pivot.iloc[1:].rolling(window=6).mean()
    
    forecast = pd.concat([monthly_percent_pivot[0:-6] ,ma.iloc[-6:-1]], axis=0).cumsum(axis=0).clip(upper=1.)
    
    ax = monthly_percent_pivot.cumsum(axis=0)[monthly_percent_pivot.columns[0:10]].plot(legend=False)
    forecast[monthly_percent_pivot.columns[0:10]][-6:].plot(ax=ax, legend=False, color='red', style=['--' for col in forecast.columns])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0%}'.format(x)))
    title = 'Moving Average 6 Month Forecast'
    plt.title(title)
    plt.savefig('files\\'+title)
