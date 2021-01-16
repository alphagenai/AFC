# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:36:22 2021

@author: mark
"""

import random
import logging
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

from calculate_days_dropped import calculate_days_dropped
from individual_analysis1 import create_percent_sdf
from calc_PD import plot_beta

""" NEED TO CALCULATE P(E| NOT IN PAR30) """

class PE_calc(object):
    def __init__(self, daily_sdf_fullts):
        no_elec = daily_sdf_fullts['elec_is_off'].unstack(0).astype('boolean')

        daily_sdf_pivot = daily_sdf_fullts['AmountPaid'].unstack(0)

        daily_cumulative_percent_sdf_pivot = create_percent_sdf(daily_sdf_pivot, 
                                                                  cumulative=True, cohort='dec_17')
        
        fully_paid = daily_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag
        
        ## completed contracts are converted to NaN
        ## be careful that the start and end dates of both dataframes is the same
        no_elec = no_elec.mask(fully_paid).astype('boolean')
        
        self.no_elec = no_elec


        par30_daily_pivot = daily_sdf_fullts['PAR30+'].unstack(0).sort_index()
    
        ## completed contracts are converted to NaN
        ## be careful that the start and end dates of both dataframes is the same
        fully_paid = daily_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag
        par30_daily_pivot = par30_daily_pivot.mask(fully_paid).astype('boolean')
        
        self.par30_daily_pivot = par30_daily_pivot 

    def update_counterparty_PE(self, counterparty, forecast_startdate):
        hist_e = ~self.no_elec.loc[:forecast_startdate, counterparty.CounterpartyId] # includes forecast_startdate
        hist_par30 = self.par30_daily_pivot.loc[:forecast_startdate, counterparty.CounterpartyId]
        
        ## WANT PE CONDITIONAL ON NOT PAR30+ !
        conditional = hist_e & ~hist_par30
        #unconditional = hist_e.groupby(hist_e).count()

        counterparty.alpha += conditional[True]
        counterparty.beta += conditional[False]


if __name__ == "__main__":
    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    df = df.groupby(['ContractId', 'TransactionTS']).sum()
    daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
    daily_sdf_pivot = daily_sdf['AmountPaid'].unstack(0).fillna(0).sort_index()
    
    daily_sdf_fullts = calculate_days_dropped(daily_sdf)
    
    daily_cumulative_percent_sdf_pivot = create_percent_sdf(daily_sdf_pivot, 
                                                              cumulative=True, cohort='dec_17')
    
    no_elec = daily_sdf_fullts['elec_is_off'].unstack(0).astype('boolean')
    
    fully_paid = daily_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag
    
    ## completed contracts are converted to NaN
    ## be careful that the start and end dates of both dataframes is the same
    no_elec = no_elec.mask(fully_paid).astype('boolean')
    
    no_elec_yesterday = no_elec.shift(1)
    
    ## ignore first month because we do not know when in the month the contract started
    
    no_elec = no_elec.loc['1/1/2018':'31/10/2020']
    no_elec_yesterday = no_elec_yesterday.loc['1/1/2018':'31/10/2020']
    
    one_contract_id = no_elec.columns[1]  # nice and switchy
    one_contract_id = no_elec.columns[3]  # same as lattice
    one_contract_id = no_elec.columns[0]  # lots of defaults



def plot_elec_clusters(forecast_startdate, one_contract_id=None):
    if one_contract_id is None:
        i = random.randint(0,1000)
        one_contract_id = no_elec.columns[i]
    
    hist_e = ~no_elec.loc[:forecast_startdate, one_contract_id]

    fig, ax = plt.subplots(1,2)
    hist_e.astype(float).plot(ax=ax[0])
    ax[0].set_title("Electricity Clustering")
    ax[0].set_xlabel("")

    unconditional = hist_e.groupby(hist_e).count()
    plot_beta(unconditional[True], unconditional[False], ax=ax[1])
    ax[1].set_title("Probability of Electricity")
    
    plt.suptitle(one_contract_id)
    plt.savefig('files\\Electricity Clustering for {}.png'.format(one_contract_id))
    return one_contract_id

def all_plots():
    interesting_contracts = ['1352353',
                             '1358791',
                             '1349722',  # same as lattice
                             '1349768', '1349968', '1350061', '1350103', '1350357',  # defaulters
                             ]
    
    for cid in interesting_contracts:
        plot_elec_clusters(cid)
    
    defaulters = default[default].iloc[10:30].index
    
    for cid in defaulters:
        try:
            plot_elec_clusters(cid)
        except KeyError:
            print("There were no examples of True")

# hist_no_e = no_elec.loc[:forecast_startdate, one_contract_id]
# hist_yest = no_elec_yesterday.loc[:forecast_startdate, one_contract_id]

# hist_df = pd.concat([hist_no_e, hist_yest], axis=1)    

