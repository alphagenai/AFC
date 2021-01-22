# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:58:14 2020

@author: mark
"""




import pandas as pd
import numpy as np

import logging

from individual_analysis1 import create_small_df, create_percent_sdf, convert_to_daily_pivot

def main():
    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    
    df = df.groupby(['ContractId', 'TransactionTS']).sum()
    
    monthly_sdf = df.groupby(['ContractId', pd.Grouper(freq='M', level=1)])['AmountPaid'].sum().to_frame()
    monthly_sdf_fullts = calc_moving_average(monthly_sdf)
    return monthly_sdf_fullts

def calc_moving_average(monthly_sdf):
    """ Monthly moving average NOT INCLUDING ZERO-PAYMENT MONTHS """
    logging.info('Calculating average monthly payments NOT INCLUDING ZERO-PAYMENT MONTHS')
    
    monthly_sdf = monthly_sdf.reorder_levels(
        ['ContractId', 'TransactionTS'])
    
    date_idx = pd.date_range('31/12/2017', '30-11-2020', freq='M')
    idx = pd.MultiIndex.from_product(
        [monthly_sdf.index.levels[0], date_idx], names=['ContractId', 'TransactionTS'])
    
    monthly_sdf_fullts = monthly_sdf.sort_index().reindex(idx,).fillna(0)    
    
    monthly_sdf_fullts['paid'] = monthly_sdf_fullts!=0
    
    ## need to drop the first month and its not currently grouping by contractID properly
     
    monthly_sdf_fullts["MovingAverage"] = monthly_sdf_fullts.reset_index(level=0).groupby(
        ['ContractId', 'paid',])['AmountPaid'].rolling(6).mean().unstack('paid')[True]
    return monthly_sdf_fullts


if __name__ == "__main__":
    monthly_sdf_fullts = main()