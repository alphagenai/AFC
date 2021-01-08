# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:36:22 2021

@author: mark
"""

import random
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from calculate_days_dropped import calculate_days_dropped
from individual_analysis1 import create_percent_sdf
from calc_PD import plot_beta

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


forecast_startdate = '2019-12-31'


def plot_elec_clusters(one_contract_id=None):
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
    
    plt.savefig('files\\Electricity Clustering for {}.png'.format(one_contract_id))
    return one_contract_id

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

hist_no_e = no_elec.loc[:forecast_startdate, one_contract_id]
hist_yest = no_elec_yesterday.loc[:forecast_startdate, one_contract_id]

hist_df = pd.concat([hist_no_e, hist_yest], axis=1)    


