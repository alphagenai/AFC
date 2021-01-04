# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 00:12:12 2020

@author: mark
"""


import pandas as pd
import numpy as np

from individual_analysis1 import create_small_df, create_percent_sdf, convert_to_daily_pivot

def main():
    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    
    df = df.groupby(['ContractId', 'TransactionTS']).sum()
    
    daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
    monthly_sdf = daily_sdf.groupby(['ContractId',pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
    monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
    PD_given_D, PD_given_ND, PND_given_D, PND_given_ND = calc_PD(monthly_sdf_pivot)


def daily_default_full_timeseries():
    default = daily_sdf_fullts['PAR30+']
    default_event = (default.diff() != 0) & (daily_sdf_fullts['PAR30+']==True)
    recovery_event = (default.diff() != 0) & (daily_sdf_fullts['PAR30+']==False) & (
        (default.index.get_level_values(1).month != 12) & (default.index.get_level_values(1).year != 2017)
        )
    daily_sdf_fullts[recovery_event].to_csv('recoveries.csv')
    daily_sdf_fullts[default_event].to_csv('defauls.csv')



#############  FREQUENTIST


def calc_PD(monthly_sdf_pivot, ):
    
    monthly_cumulative_percent_sdf_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=True, cohort='dec_17')
    
    
    first_payment = monthly_sdf_pivot.iloc[1]
    
    first_payment[first_payment==0]
    
    
    
    ## 1) MONTH OF DEFAULT DOES NOT MATTER
    
    
    defaults = monthly_sdf_pivot.iloc[1:]==0 # total non-NaN: 28,593
    fully_paid = monthly_cumulative_percent_sdf_pivot >= 0.99
    
    ## completed contracts are converted to NaN
    defaults = defaults.mask(fully_paid).astype('boolean')
    
    individual_uncond_PD = defaults.sum(axis=0, skipna=True)/defaults.count(axis=0)  #df.count() does not count NaNs
    unconditional_PD = defaults.sum(axis=0, skipna=True).sum()/defaults.count(axis=0).sum()
    
    last_month_defaults = defaults.shift(1)
    
    D_given_D = defaults & last_month_defaults  # 2,233 events
    D_given_ND = defaults & ~last_month_defaults # 1,139
    
    ND_given_D = ~defaults & last_month_defaults # 1,060
    ND_given_ND = ~defaults & ~last_month_defaults #23,161
    
    
    PD_given_D = D_given_D.sum().sum() / D_given_D.count().sum()  #2,233/29,024
    PD_given_ND = D_given_ND.sum().sum() / D_given_ND.count().sum()
    
    PND_given_D = ND_given_D.sum().sum() / ND_given_D.count().sum() #1,060/28,198
    PND_given_ND = ND_given_ND.sum().sum() / ND_given_ND.count().sum()
    
    ## for completeness
    paid_off_months = fully_paid.sum().sum() # 6,407
    P_paid_off = paid_off_months / monthly_sdf_pivot.count().sum()  ## 6,407/36,000
    
    
    ## why doesnt this equal 1
    PD_given_D + PD_given_ND + PND_given_D + PND_given_ND
    
    np.sum([D_given_D.sum().sum(),
        D_given_ND.sum().sum(),
        ND_given_D.sum().sum(),
        ND_given_ND.sum().sum(),
        paid_off_months],)  #34,000 ==> 34 months ==> 2 months not included
    
    PD_dict = {'PD_given_D':PD_given_D, 
               'PD_given_ND':PD_given_ND, 
               'PND_given_D':PND_given_D, 
               'PND_given_ND':PND_given_ND,
               }
    return PD_dict
    
    ## 2) MONTH MATTERS
    
def calc_monthly_PD():
    PD_given_D = D_given_D.sum(axis=1) / D_given_D.count(axis=1)
    PD_given_ND = D_given_ND.sum(axis=1) / D_given_ND.count(axis=1)
    
    PND_given_D = ND_given_D.sum(axis=1)/ ND_given_D.count(axis=1)
    PND_given_ND = ND_given_ND.sum(axis=1)/ ND_given_ND.count(axis=1)
    
    ## also dont always sum to 1
    PD_given_D + PD_given_ND + PND_given_D + PND_given_ND




#### BAYESIAN

## TO DO


if __name__ == "__main__":
    main()