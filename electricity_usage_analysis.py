# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:29:32 2021

@author: mark
"""

import pandas as pd
import seaborn as sns
from basic_datasets import BasicDatasets

def first_analysis(daily_sdf_fullts):
    monthly_days_of_elec = (~daily_sdf_fullts['elec_is_off']).groupby(['ContractId', pd.Grouper(freq='1M', level=1)]).sum()
    
    # df = elec_off.to_frame()
    # df['TransactionTS'] = df.index.get_level_values(1)
    
    # df['days_in_month'] = df.TransactionTS.apply(lambda x: x.days_in_month)
    
    sns.histplot(monthly_days_of_elec, bins=31)
    plt.title('Days of electricity used per month')
    plt.savefig('files\\Days of electricity used per month.png')
    
    ## need to make sure that both these time series are in daily frequency
    no_elec_pivot = daily_sdf_fullts['days_out_of_elec'].unstack(0).loc['2017-12-01':'2020-11-17'] #.mask(fully_paid)
    ## days without elec at end of ts
    noelec_end = no_elec_pivot.iloc[-1]
    
    ## max days without elec
    noelec_max = no_elec_pivot.max()
    
    (noelec_end > 30).sum()
    ((noelec_end == noelec_max) & (noelec_end >30)).sum()
    
if __name__ == "__main__":
    bd = BasicDatasets()
    dfts = bd.daily_ts #full dataset excluding paid off contracts
    
    monthly_elec_pivot = monthly_days_of_elec.unstack(0)
    monthly_elec_pivot.mean(axis=0)
    
    monthly_elec_pivot.divide(monthly_elec_pivot.index.daysinmonth, axis=0)