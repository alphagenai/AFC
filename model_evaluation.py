# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:07:19 2021

@author: mark
"""


import pandas as pd
import numpy as np


from LatticeModel import LatticeModel
from calc_PD import PDCalculator
from monthly_averages import calc_moving_average

from scipy.stats import beta

def initialise():
    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    
    monthly_sdf = df.groupby(['ContractId',pd.Grouper(key='TransactionTS', freq='M',)])['AmountPaid'].sum()
    monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
    
    pd_calc = PDCalculator(monthly_sdf_pivot)
    PD_dict = pd_calc.calc_PD(monthly_sdf_pivot)
    
    monthly_sdf_fullts = calc_moving_average(monthly_sdf.to_frame())
    
    ma_pivot = monthly_sdf_fullts['MovingAverage'].unstack(0).shift(1).fillna(method='ffill') #shift(1) for next month forecast, ffill for future months with no MA (because no payments made)
    
    return monthly_sdf_pivot, ma_pivot, PD_dict

#one_ma = ma_pivot[one_contract_id]
#forecast_dates = pd.date_range('2019-6-30', '2019-11-30', freq='1M')

monthly_sdf_pivot, ma_pivot, PD_dict = initialise()
for cid in monthly_sdf_pivot.columns[1:3]:
    one_ma = ma_pivot[cid]
    #for forecast_startdate in forecast_dates:

    def six_month_expectation(cid, one_ma, monthly_sdf_pivot, PD_dict):
        forecast_startdate = '2019-6-30'
        average_payment = one_ma[forecast_startdate]
        initial_payment = monthly_sdf_pivot.loc[forecast_startdate, cid] 

        lm = LatticeModel(initial_payment, average_payment=average_payment, 
                          contract_id=cid, forecast_startdate=forecast_startdate, 
                          PD_dict=PD_dict)


        for t in range(6):
            lm.add_level()
        print(lm.calculate_expectation(t=6))



