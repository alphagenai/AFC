# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:07:19 2021

@author: mark
"""

import time

import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.stats import beta
import matplotlib.ticker as mtick

from LatticeModel import LatticeModel
from calc_PD import PDCalculator
from monthly_averages import calc_moving_average
from Counterparty import Counterparty
from individual_analysis1 import create_percent_sdf



"""
TO DO: SORT OUT WHAT HAPPENS WITH NAN
"""

def initialise(use_values=True):
    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    
    monthly_sdf = df.groupby(['ContractId',pd.Grouper(key='TransactionTS', freq='M',)])['AmountPaid'].sum()
    monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
        
    monthly_perc_sdf_pivot = create_percent_sdf(monthly_sdf_pivot,
                                                cumulative=False, use_monthdiff=False, cohort='dec_17')
    
    if use_values:
        monthly_sdf_fullts = calc_moving_average(monthly_sdf.to_frame('AmountPaid'))
    else:
        monthly_sdf_fullts = calc_moving_average(monthly_perc_sdf_pivot.stack().to_frame('AmountPaid'))
    ma_pivot = monthly_sdf_fullts['MovingAverage'].unstack(0).shift(1).fillna(method='ffill') #shift(1) for next month forecast, ffill for future months with no MA (because no payments made)
    
    if use_values:
        return monthly_sdf_pivot, ma_pivot
    else:
        return monthly_perc_sdf_pivot, ma_pivot

def six_month_expectation(counterparty, ma_pivot, monthly_sdf_pivot, 
                          pd_calc, forecast_startdate):
    forecast_startdate = '2019-6-30'
    average_payment = ma_pivot.loc[forecast_startdate, counterparty.ContractId]
    initial_payment = monthly_sdf_pivot.loc[forecast_startdate, counterparty.ContractId] 

    counterparty.update_bayesian_mean_PDs(monthly_sdf_pivot, forecast_startdate, pd_calc)

    lm = LatticeModel(initial_payment, average_payment=average_payment, 
                      contract_id=counterparty.ContractId, 
                      forecast_startdate=forecast_startdate, 
                      PD_dict=counterparty.PD_dict)


    for t in range(6):
        lm.add_level()
    return lm, lm.calculate_expectation(t=6)


if __name__ == "__main__":
    monthly_sdf_pivot, ma_pivot = initialise(use_values=False)
    forecast_startdate = '2019-06-01'
    
    pd_calc = PDCalculator(monthly_sdf_pivot)
    _, _ = pd_calc.data_prep()
    
    counterparty_dict = {}
    diff_list = []
    for cid in monthly_sdf_pivot.columns:
        tic = time.perf_counter()
        counterparty_dict[cid] = Counterparty(cid)
        #for forecast_startdate in forecast_dates:
        lm, counterparty_dict[cid].six_month_expectation = six_month_expectation(
            counterparty_dict[cid], ma_pivot, monthly_sdf_pivot, 
            pd_calc,  forecast_startdate)  

        actual = monthly_sdf_pivot.loc[
            forecast_startdate:pd.Timestamp(forecast_startdate)+pd.DateOffset(months=6), counterparty_dict[cid].ContractId
                                       ].cumsum().iloc[-1]

        diff_list += [counterparty_dict[cid].six_month_expectation - actual,]
        toc = time.perf_counter()
        print(f"Time taken to forecast one counterparty: {toc - tic:0.4f} seconds")
        
    sns.histplot(diff_list)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.title('Histogram of Model Error')
    plt.savefig('files\\histogram of errors.png')