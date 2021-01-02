# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:10:27 2020

@author: mark
"""

import matplotlib.ticker as ticker

from matplotlib import pyplot as plt

ma = monthly_percent_pivot.iloc[1:]['AmountPaid'].rolling(window=6).mean()

forecast = pd.concat([monthly_percent_pivot[0:-6] ,ma.iloc[-6:-1]], axis=0).cumsum(axis=0).clip(upper=1.)

ax = monthly_percent_pivot.cumsum(axis=0)[monthly_percent_pivot.columns[0:10]].plot(legend=False)
forecast[monthly_percent_pivot.columns[0:10]][-6:].plot(ax=ax, legend=False, color='red', style=['--' for col in forecast.columns])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0%}'.format(x)))
title = 'Moving Average 6 Month Forecast'
plt.title(title)
plt.savefig('files\\'+title)
