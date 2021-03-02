# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:07:19 2021

@author: mark
"""

import time
import logging
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick

from matplotlib import pyplot as plt
from scipy.stats import beta
from sklearn.metrics import mean_squared_error
from datetime import datetime



from LatticeModel import LatticeModel
from calc_PD import PDCalculator
from monthly_averages import calc_moving_average
from Counterparty import Counterparty
from individual_analysis1 import create_percent_sdf
from moving_average_model import six_month_ma
from basic_datasets import BasicDatasets


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
    average_payment = ma_pivot.loc[forecast_startdate, counterparty.ContractId]
    logging.debug('Average payment: {}'.format(average_payment))
    initial_payment = monthly_sdf_pivot.loc[forecast_startdate, counterparty.ContractId] 
    logging.debug('Initial payment: {}'.format(initial_payment))
    paid_so_far = monthly_sdf_pivot.loc[:forecast_startdate, counterparty.ContractId].sum()
    logging.debug('Paid so far: {}'.format(paid_so_far))

    if np.isnan(average_payment):
        logging.warning("Not enough payment data to calculate moving average")    
        average_payment = paid_so_far.mean()

    counterparty.update_bayesian_mean_PDs(monthly_sdf_pivot, forecast_startdate, pd_calc)
    logging.debug('counterparty params updated to {}'.format(counterparty.params))

    lm = LatticeModel(initial_payment, paid_so_far=paid_so_far, average_payment=average_payment, 
                      contract_id=counterparty.ContractId, 
                      forecast_startdate=forecast_startdate, 
                      PD_dict=counterparty.PD_dict)


    for t in range(6):
        lm.add_level()
    return lm, lm.calculate_expectation(t=6)



def eval_one_counterparty(cparty, ma_pivot, monthly_sdf_pivot, 
        pd_calc, forecast_startdate, diff_list, results_dict, ew):
    tic = time.perf_counter()
    
    paid_so_far = monthly_sdf_pivot.loc[:forecast_startdate, cparty.ContractId].sum()
    logging.debug('Date: {}, CID: {}, Paid so far: {}'.format(forecast_startdate, cparty.ContractId, paid_so_far))
    
    lm, cparty.six_month_expectation = six_month_expectation(
        cparty, ma_pivot, monthly_sdf_pivot, 
        pd_calc,  forecast_startdate)  
    lm.df.to_excel(
        ew, sheet_name='Vals-{}'.format(forecast_startdate), startrow=ew.next_row)
    cparty.PD_df.to_excel(
        ew, sheet_name='PDs-{}'.format(forecast_startdate), startrow=ew.next_row)
    ew.next_row += lm._df.shape[0]+2
    logging.debug('df size {}, max t: {}, next_row: {}'.format(lm._df.shape[0],lm._df.index.max() , ew.next_row))
    
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

def plot_errors(diff_list, title=None, forecast_startdate=None):
    fig, ax = plt.subplots()
    sns.histplot(diff_list)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlim(-0.3, 0.3)
    plt.ylim(0, 150)

    if title is None:
        title = 'Histogram of Model Error'
    plt.title(title)
    plt.savefig('files\\{}_{}.png'.format(title, forecast_startdate))
    

    
if __name__ == "__main__":
    
    logfilename=r'files\\logfiles\\model_eval_{}.log'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
    logging.basicConfig(filename='files\\logfiles\\model_evaluation.log', level=logging.DEBUG,)
    monthly_sdf_pivot, ma_pivot = initialise(use_values=False)  #use_values=True is shilling values, False is percentage of TCV
    ma_pivot_incl0 = monthly_sdf_pivot.iloc[1:].rolling(window=6).mean().shift(1)

    #forecast_startdate = '2019-06-01'
    #forecast_dates = pd.date_range('2019-04-30', '2019-12-31', freq='1M')    
    forecast_dates = ['30-Jun-2020', '31-Oct-2020']

    pd_calc = PDCalculator(monthly_sdf_pivot)
    _, _ = pd_calc.data_prep()
    
    counterparty_dict = {}
    diff_list = []
    results_dict = {}

    ew = pd.ExcelWriter('files\\LatticeResults.xlsx', )

    for forecast_startdate in forecast_dates:
        ew.next_row = 0

        results_dict[forecast_startdate] = {}
        for cid in monthly_sdf_pivot.columns:
            counterparty_dict[cid] = Counterparty(cid)
            diff_list, results_dict[forecast_startdate] = eval_one_counterparty(counterparty_dict[cid], 
                                                            ma_pivot, monthly_sdf_pivot, 
                                                            pd_calc,  forecast_startdate, 
                                                            diff_list, results_dict[forecast_startdate], 
                                                            ew)

        one_rd = results_dict[forecast_startdate]
        df = pd.DataFrame.from_dict(one_rd, orient='index')
        not_completed = df[df[0]<1.0]
        
        lattice_model_error = not_completed[0] - not_completed[1]
        naive_model_error = not_completed[0] - not_completed[2]
        plot_errors(lattice_model_error , 'Histogram of Errors - Lattice Model', forecast_startdate)
        plot_errors(naive_model_error , 'Histogram of Errors - Naive Moving Average', forecast_startdate)    
        
        print('Lattice model mean error: {:.2f}'.format(lattice_model_error.mean()))
        print('Lattice model error std: {:.2f}'.format(lattice_model_error.std()))
        print('Lattice model mean squared error: {:.2f}'.format(mean_squared_error(not_completed[0],not_completed[1])))
    
        print('Naive model mean error: {:.2f}'.format(naive_model_error.mean()))
        print('Naive model error std: {:.2f}'.format(naive_model_error.std()))
        print('Naive model mean squared error: {:.2f}'.format(mean_squared_error(not_completed[0],not_completed[2])))

    ew.close()



    
def results_checker():    
    for cid in results_dict['30-Jun-2020'].keys():
        lm1 = results_dict['30-Jun-2020'][cid][-1]
        
        lm2 = results_dict['31-Oct-2020'][cid][-1]
        print((lm1.df == lm2.df).all().all())
        
        
        
    for cid, cpty in counterparty_dict.items():
        print(cpty.params)