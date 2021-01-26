# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:07:19 2021

@author: mark
"""

import time
import logging

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick

from matplotlib import pyplot as plt
from scipy.stats import beta
from sklearn.metrics import mean_squared_error


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
        
    
    ma_pivot = monthly_sdf_fullts['MovingAverage'].unstack(
        0).shift(                           # shift(1) for next month forecast
            1).fillna(                      # NO LONGER NECESSARY ffill for future months with no MA (because no payments made)
            method='ffill').fillna(0)       # fillna(0) for contracts with not enough payments to create a moving average
    
    if use_values:
        return monthly_sdf_pivot, ma_pivot
    else:
        return monthly_perc_sdf_pivot, ma_pivot

def six_month_expectation(counterparty, ma_pivot, monthly_sdf_pivot, 
                          pd_calc, forecast_startdate):
    forecast_startdate = '2019-6-30'
    average_payment = ma_pivot.loc[forecast_startdate, counterparty.ContractId]
    initial_payment = monthly_sdf_pivot.loc[forecast_startdate, counterparty.ContractId] 
    paid_so_far = monthly_sdf_pivot.loc[:forecast_startdate, counterparty.ContractId].sum()

    if np.isnan(average_payment):
        logging.warning("Not enough payment data to calculate moving average")    
        average_payment = paid_so_far.mean()

    counterparty.update_bayesian_mean_PDs(monthly_sdf_pivot, forecast_startdate, pd_calc)

    lm = LatticeModel(initial_payment, paid_so_far=paid_so_far, average_payment=average_payment, 
                      contract_id=counterparty.ContractId, 
                      forecast_startdate=forecast_startdate, 
                      PD_dict=counterparty.PD_dict)


    for t in range(6):
        lm.add_level()
    return lm, lm.calculate_expectation(t=6)


def six_month_ma(counterparty, forecast_startdate,  ma_pivot_incl0):
    
        return ma_pivot_incl0.loc[forecast_startdate, counterparty.ContractId]

def eval_one_counterparty(cparty, ma_pivot, monthly_sdf_pivot, 
        pd_calc, forecast_startdate, diff_list, results_dict):
    tic = time.perf_counter()
    
    paid_so_far = monthly_sdf_pivot.loc[:forecast_startdate, cparty.ContractId].sum()
    logging.debug('Date: {}, CID: {}, Paid so far: {}'.format(forecast_startdate, cparty.ContractId, paid_so_far))
    
    lm, cparty.six_month_expectation = six_month_expectation(
        cparty, ma_pivot, monthly_sdf_pivot, 
        pd_calc,  forecast_startdate)  
    
    logging.debug('Lattice model 6 month expectation: {}'.format(cparty.six_month_expectation))
    actual = monthly_sdf_pivot.loc[
        :pd.Timestamp(forecast_startdate)+pd.DateOffset(months=6), counterparty_dict[cid].ContractId
                                   ].sum()
    logging.debug('Actual realisation: {}'.format(actual))
    
    ## to do: compare to previous code
    naive_forecast = np.min([1.0, paid_so_far + 6*six_month_ma(cparty, forecast_startdate, ma_pivot_incl0)])
    logging.debug('naive forecast: {}'.format(naive_forecast))

    
    diff_list += [cparty.six_month_expectation - actual,]
    results_dict[cid] = (actual, cparty.six_month_expectation, naive_forecast, lm)
    toc = time.perf_counter()
    logging.debug(f"Time taken to forecast one counterparty: {toc - tic:0.4f} seconds")
    
    if np.isnan(actual):
        logging.warning('Actual value for {} is nan'.format(cid))
    if np.isnan(cparty.six_month_expectation):
        logging.warning('Model value for {} is nan'.format(cid))
    return diff_list, results_dict

def plot_errors(diff_list, title=None):
    fig, ax = plt.subplots()
    sns.histplot(diff_list)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if title is None:
        title = 'Histogram of Model Error'
        plt.title(title)
    plt.savefig('files\\{}.png'.format(title))
    

    
if __name__ == "__main__":
    
    logging.basicConfig(log_level=logging.INFO)
    monthly_sdf_pivot, ma_pivot = initialise(use_values=False)  #use_values=True is shilling values, False is percentage of TCV
    ma_pivot_incl0 = monthly_sdf_pivot.iloc[1:].rolling(window=6).mean().shift(1)

    #forecast_startdate = '2019-06-01'
    forecast_dates = pd.date_range('2019-04-30', '2019-12-31', freq='1M')    


    pd_calc = PDCalculator(monthly_sdf_pivot)
    _, _ = pd_calc.data_prep()
    
    counterparty_dict = {}
    diff_list = []
    results_dict = {}
    for forecast_startdate in forecast_dates:
        results_dict[forecast_startdate] = {}
        for cid in monthly_sdf_pivot.columns:
            counterparty_dict[cid] = Counterparty(cid)
            diff_list, results_dict[forecast_startdate] = eval_one_counterparty(counterparty_dict[cid], 
                                                            ma_pivot, monthly_sdf_pivot, 
                                                            pd_calc,  forecast_startdate, 
                                                            diff_list, results_dict[forecast_startdate])


    one_rd = results_dict[forecast_dates[0]]
    df = pd.DataFrame.from_dict(one_rd, orient='index')
    plot_errors(df[0] - df[1], 'Histogram of Errors - Lattice Model')
    plot_errors(df[0] - df[2], 'Histogram of Errors - Naive Moving Average')    
    
    cid = '1354267'