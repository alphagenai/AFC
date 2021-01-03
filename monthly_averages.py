# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:58:14 2020

@author: mark
"""




import pandas as pd
import numpy as np

from individual_analysis1 import create_small_df, create_percent_sdf, convert_to_daily_pivot

def main():
    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    
    df = df.groupby(['ContractId', 'TransactionTS']).sum()
    
    monthly_sdf = df.groupby(['ContractId', pd.Grouper(freq='M', level=1)]).sum()
    calc_moving_average(monthly_sdf)

def calc_moving_average(monthly_sdf):

    date_idx = pd.date_range('31/12/2017', '30-11-2020', freq='M')
    idx = pd.MultiIndex.from_product(
        [monthly_sdf.index.levels[0], date_idx], names=['ContractId', 'TransactionTS'])
    
    monthly_sdf_fullts = monthly_sdf.reindex(idx,).fillna(0)    
    
    monthly_sdf_fullts['paid'] = monthly_sdf_fullts['AmountPaid']!=0
    
    ## need to drop the first month and its not currently grouping by contractID properly
     
    monthly_sdf_fullts["MovingAverage"] = monthly_sdf_fullts.groupby(['ContractId', 'paid',])['AmountPaid'].rolling(6).mean().unstack(1)[True].droplevel(level=0)
    return monthly_sdf_fullts


if __name__ == "__main__":
    main()