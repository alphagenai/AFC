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
    return monthly_sdf_pivot


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

def calc_PD(monthly_sdf_pivot, point_estimate=True):
    
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

    if point_estimate:
        PD_dict = point_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND)
    return PD_dict
    
def point_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND): 
    # PD_given_D = D_given_D.sum().sum() / D_given_D.count().sum()  #2,233/28,647
    # PD_given_ND = D_given_ND.sum().sum() / D_given_ND.count().sum() # 1,139/28,124
    
    # PND_given_D = ND_given_D.sum().sum() / ND_given_D.count().sum() #1,060/28,647
    # PND_given_ND = ND_given_ND.sum().sum() / ND_given_ND.count().sum()  #23,161/28,124
    ## Total from Excel: 27,593

    total_given_D = how_many_true(D_given_D) + how_many_true(ND_given_D)
    total_given_ND = how_many_true(D_given_ND) + how_many_true(ND_given_ND)

    PD_given_D = how_many_true(D_given_D) / total_given_D 
    PD_given_ND = how_many_true(D_given_ND) / total_given_ND # = (1 - PD_given_D)
    PND_given_D = how_many_true(ND_given_D) / total_given_D # = (1 - PND_given_ND)
    PND_given_ND = how_many_true(ND_given_ND) / total_given_ND 

    logging.debug('PD_given_D : {} / {}'.format(how_many_true(D_given_D) , total_given_D))    
    logging.debug('PD_given_ND : {} / {}'.format(how_many_true(D_given_ND) , total_given_ND))    
    logging.debug('PND_given_D : {} / {}'.format(how_many_true(ND_given_D) , total_given_D))    
    logging.debug('PND_given_ND : {} / {}'.format(how_many_true(ND_given_ND) , total_given_ND))    

   
    PD_dict = {'PD_given_D':PD_given_D, 
               'PD_given_ND':PD_given_ND, 
               'PND_given_D':PND_given_D, 
               'PND_given_ND':PND_given_ND,
               }
    return PD_dict
    
    ## 2) MONTH MATTERS
    
def temporal_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND):

    total_given_D = D_given_D.sum(axis=1) + ND_given_D.sum(axis=1)
    total_given_ND = D_given_ND.sum(axis=1) + ND_given_ND.sum(axis=1)

    

    PD_given_D = D_given_D.sum(axis=1) / total_given_D 
    PD_given_ND = D_given_ND.sum(axis=1) / total_given_ND # = (1 - PD_given_D)
    PND_given_D = ND_given_D.sum(axis=1) / total_given_D # = (1 - PND_given_ND)
    PND_given_ND = ND_given_ND.sum(axis=1) / total_given_ND 
    
    PD_given_D + PD_given_ND + PND_given_D + PND_given_ND
    PD_dict = {'PD_given_D':PD_given_D, 
               'PD_given_ND':PD_given_ND, 
               'PND_given_D':PND_given_D, 
               'PND_given_ND':PND_given_ND,
               }
    return PD_dict


def counterparty_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND):

    total_given_D = D_given_D.sum(axis=0) + ND_given_D.sum(axis=0)
    total_given_ND = D_given_ND.sum(axis=0) + ND_given_ND.sum(axis=0)

    

    PD_given_D = D_given_D.sum(axis=0) / total_given_D 
    PD_given_ND = D_given_ND.sum(axis=0) / total_given_ND # = (1 - PD_given_D)
    PND_given_D = ND_given_D.sum(axis=0) / total_given_D # = (1 - PND_given_ND)
    PND_given_ND = ND_given_ND.sum(axis=0) / total_given_ND 
    
    PD_given_D + PD_given_ND + PND_given_D + PND_given_ND
    PD_dict = {'PD_given_D':PD_given_D, 
               'PD_given_ND':PD_given_ND, 
               'PND_given_D':PND_given_D, 
               'PND_given_ND':PND_given_ND,
               }
    return PD_dict



#### BAYESIAN

def label_logic(two_element_series):
    if two_element_series[0] & two_element_series[1]:
        return 1
    if two_element_series[0] & ~two_element_series[1]:
        return 2
    if ~two_element_series[0] & two_element_series[1]:
        return 3
    if ~two_element_series[0] & ~two_element_series[1]:
        return 4
    
def prior_analysis():
    c_PD_dict = counterparty_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND)
    t_PD_dict = temporal_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND)

    cdf = pd.DataFrame(c_PD_dict)
    tdf = pd.DataFrame(t_PD_dict)
    cdf = cdf[['PD_given_D', 'PD_given_ND']]
    tdf = tdf[['PD_given_D', 'PD_given_ND']]
    

    df = df[~df.isna().any(axis=1)]
    #df = df[~(df==1.).any(axis=1)]
    #df = df[~(df==0.0).any(axis=1)]

    df = pd.concat([cdf[['PD_given_D', 'PD_given_ND']],
                  tdf[['PD_given_D', 'PD_given_ND']],]
                  axis=0)


    df.plot(kind='hist', bins=50)

def bayes_PD_given_D_update(alpha, beta, label):
    """ This is where a HMM could come in - classify customers into credit buckets for conditional PDs """

    if outcome:
        alpha += 1
    else:
        beta +=1

    #PD_given_D_prior = Beta(alpha,beta)
    #likelihood = Bernoulli(p)
    #posterior = Beta(7+1, 5) or Beta(7,5+1)
    
    new_mean = alpha / (alpha+beta)
    
    return alpha, beta

if __name__ == "__main__":
    monthly_sdf_pivot = main()
    #PD_dict = calc_PD(monthly_sdf_pivot)
    
    one_contract_id = monthly_sdf_pivot.columns[1]  # nice and switchy
    forecast_startdate = '2019-12-31'

    hist_d = defaults.loc[:forecast_startdate, one_contract_id]
    hist_prev_d = last_month_defaults.loc[:forecast_startdate, one_contract_id]

    hist_df = pd.concat([hist_d, hist_prev_d], axis=1)    

    print(hist_df)
    
    labels = hist_df.apply(label_logic, axis=1)
    