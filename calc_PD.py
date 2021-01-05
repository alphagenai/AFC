# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 00:12:12 2020

@author: mark
"""


import pandas as pd
import numpy as np
import logging

from individual_analysis1 import create_percent_sdf

import sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(message)s')

def main():

    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    
    df = df.groupby(['ContractId', 'TransactionTS']).sum()
    
    daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
    monthly_sdf = daily_sdf.groupby(['ContractId',pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
    monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
    PD_dict = calc_PD(monthly_sdf_pivot)


def daily_default_full_timeseries():
    default = daily_sdf_fullts['PAR30+']
    default_event = (default.diff() != 0) & (daily_sdf_fullts['PAR30+']==True)
    recovery_event = (default.diff() != 0) & (daily_sdf_fullts['PAR30+']==False) & (
        (default.index.get_level_values(1).month != 12) & (default.index.get_level_values(1).year != 2017)
        )
    daily_sdf_fullts[recovery_event].to_csv('recoveries.csv')
    daily_sdf_fullts[default_event].to_csv('defaults.csv')



#############  FREQUENTIST

def how_many_true(df):
    return df.sum().sum()

def how_many(df):
    return df.count().sum()

def calc_PD(monthly_sdf_pivot, ):
    
    monthly_cumulative_percent_sdf_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=True, cohort='dec_17')
    

    ## 1) MONTH OF DEFAULT DOES NOT MATTER
    
    
    defaults = (monthly_sdf_pivot==0).astype('boolean') # total non-NaN: 36,000 incl Dec; 28,593 incl. Jan 18
    paid  = (monthly_sdf_pivot!=0).astype('boolean')
    fully_paid = monthly_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag
    
    ## completed contracts are converted to NaN
    defaults = defaults.mask(fully_paid).astype('boolean')
    paid  = paid.mask(fully_paid).astype('boolean')
    
    #individual_uncond_PD = defaults.sum(axis=0, skipna=True)/defaults.count(axis=0)  #df.count() does not count NaNs
    #unconditional_PD = defaults.sum(axis=0, skipna=True).sum()/defaults.count(axis=0).sum()
    
    last_month_defaults = defaults.shift(1)

    ## ignore first 2 months as cannot default in origination month
    defaults = defaults.iloc[2:]  # 3,372 defaults in 28,214 total 
    last_month_defaults = last_month_defaults.iloc[2:]  #28,647
    paid = paid.iloc[2:] #24,752 of 28,124

    """ TO DO: FIX THIS: """
    ### REMEMBER FALSE & NA = FALSE!!
    D_given_D = defaults & last_month_defaults  # 2,233 events
    D_given_ND = defaults & ~last_month_defaults # 1,139
    
    
    ND_given_D = ~defaults & last_month_defaults # 1,073
    ND_given_ND = ~defaults & ~last_month_defaults # 23,679

    NA_given_D = defaults.isna() & last_month_defaults # 0
    NA_given_ND = defaults.isna() & ~last_month_defaults # 523 ##FIX THIS

    ## for completeness
    D_given_NA = defaults & last_month_defaults.isna() # 0
    ND_given_NA = ~defaults & last_month_defaults.isna() # 0 


    PD_dict = point_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND)
    return PD_dict
    
def point_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND): 
    PD_given_D = D_given_D.sum().sum() / D_given_D.count().sum()  #2,233/28,647
    PD_given_ND = D_given_ND.sum().sum() / D_given_ND.count().sum() # 1,139/28,124
    
    PND_given_D = ND_given_D.sum().sum() / ND_given_D.count().sum() #1,060/28,647
    PND_given_ND = ND_given_ND.sum().sum() / ND_given_ND.count().sum()  #23,161/28,124
    ## Total from Excel: 27,593
    
    """    
    ## for completeness
    paid_off_months = fully_paid.sum().sum() # 6,407
    P_paid_off = paid_off_months / monthly_sdf_pivot.count().sum()  ## 6,407/36,000
    
    
    ## why doesnt this equal 1
    PD_given_D + PD_given_ND + PND_given_D + PND_given_ND
    
    np.sum([D_given_D.sum().sum(),
        D_given_ND.sum().sum(),
        ND_given_D.sum().sum(),
        ND_given_ND.sum().sum(),  # 27,593 same as Excel 
        paid_off_months],)  #34,000 ==> 34 months ==> 2 months not included
    """
    
    PD_dict = {'PD_given_D':PD_given_D, 
               'PD_given_ND':PD_given_ND, 
               'PND_given_D':PND_given_D, 
               'PND_given_ND':PND_given_ND,
               }
    return PD_dict
    
    ## 2) MONTH MATTERS
    
def temporal_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND):
    PD_given_D = D_given_D.sum(axis=1) / D_given_D.count(axis=1)  ## gets larger as time goes on as good payers have finished their contracts
    PD_given_ND = D_given_ND.sum(axis=1) / D_given_ND.count(axis=1)
    
    PND_given_D = ND_given_D.sum(axis=1)/ ND_given_D.count(axis=1)
    PND_given_ND = ND_given_ND.sum(axis=1)/ ND_given_ND.count(axis=1)
    
    ## also dont always sum to 1
    PD_given_D + PD_given_ND + PND_given_D + PND_given_ND
    PD_dict = {'PD_given_D':PD_given_D, 
               'PD_given_ND':PD_given_ND, 
               'PND_given_D':PND_given_D, 
               'PND_given_ND':PND_given_ND,
               }
    return PD_dict




#### BAYESIAN

## TO DO

#prior = 

if __name__ == "__main__":
    main()