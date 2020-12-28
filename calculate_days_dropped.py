# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:23:07 2020

@author: mark
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from individual_analysis1 import create_small_df, PAYMENT_DATA_SQL

import google_sa_auth


"""
To dO, lets investigate the r/ship between sum(days dropped) and (total [%?] paid)
then build a classification model, then later a time series model for the worse contracts
each classification can have different prior for bayes
"""


def calculate_days_dropped(daily_sdf):



    
    daily_sdf['TransactionTS'] = daily_sdf.index.get_level_values(1)

    daily_sdf[['prev_payment_date', 'prev_duration']] = daily_sdf.groupby(level=0)[['TransactionTS', 'Duration']].shift(1)
    
    ## days since token dropped
    # WHAT HAPPENS TO ADJUSTMENTS
    
    ### this is not quite right - doesnt take into account credits unused from before
    daily_sdf['days_dropped'] = (daily_sdf['TransactionTS']  - daily_sdf['prev_payment_date'] - pd.to_timedelta(daily_sdf['prev_duration'], unit='D')).dt.days

    
    ## stupid americans
    date_idx = pd.date_range('12/01/2017', '30-11-2020')
    idx = pd.MultiIndex.from_product(
        [daily_sdf.index.levels[0], date_idx], names=['ContractId', 'TransactionTS'])

    daily_sdf_fullts = daily_sdf.reindex(idx,).fillna(0)  
    daily_sdf_fullts['elec_transaction'] = daily_sdf_fullts['Duration'] - 1

    # daily_sdf_fullts['switch'] = daily_sdf_fullts['cumsum'].mask(
    #     (daily_sdf_fullts['cumsum'] < 0) & (daily_sdf_fullts['Duration'] > 0),
    #     other = daily_sdf_fullts['Duration']
    #     )

    #daily_sdf_fullts['ffill'] = daily_sdf_fullts['switch'].ffill()
    #daily_sdf_fullts['final'] = daily_sdf_fullts['cumsum'].sub(daily_sdf_fullts['ffill'], fill_value=0)
    daily_sdf_fullts['elec_transaction_cumsum'] = daily_sdf_fullts.groupby(level=0)['elec_transaction'].apply(
        lambda x:cumsum_limit(x,floor=0)
        )


    ## this timeseries will start before contract start date
    daily_sdf_fullts['elec_is_off'] = (daily_sdf_fullts['elec_transaction_cumsum']==0) & (daily_sdf_fullts['Duration']==0)

    ## Looks complicated but its just cumsumming the Truth values, resetting when it hits a False
    daily_sdf_fullts['days_out_of_elec'] = daily_sdf_fullts.groupby(
        daily_sdf_fullts['elec_is_off'].diff().cumsum()
        )['elec_is_off'].cumsum().mask(
            ~daily_sdf_fullts['elec_is_off'], other=0
            )
    daily_sdf_fullts['days_out_of_elec2'] = daily_sdf_fullts.groupby(
        (~daily_sdf_fullts['elec_is_off']).cumsum()
        )['elec_is_off'].cumsum()
    return daily_sdf_fullts


def cumsum_limit(s, floor=-np.inf, limit=np.inf):
    out = []
    runsum = 0
    for i, v in s.iteritems():
        runsum += v
        if runsum <= floor:
            runsum = floor
        if runsum >= limit:
            runsum = limit
        out.append(runsum)
    return pd.Series(data=out, index=s.index)
    
    



if __name__ == "__main__":

    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)

    df = df.groupby(['ContractId', 'TransactionTS']).sum()

    daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()


    daily_sdf_fullts = calculate_days_dropped(daily_sdf)
    daily_sdf_fullts.to_csv('temp.csv')




def analysis_of_dropped_importance():    
    total_paid = daily_sdf.groupby('ContractId').sum()
    monthly_sdf = daily_sdf.groupby(['ContractId', pd.Grouper(freq='M', level=1)]).sum()
    monthly_cumsum = monthly_sdf.groupby('ContractId').cumsum(axis=0)
    six_month_mark = monthly_cumsum.loc(axis=0)[:,'2019-06-30 00:00:00+00:00'] ##be careful - we lose some contracts that havent paid in June
    
    contract_sql = """
        SELECT c.ContractId,
                c.MainApplicantGender, 
                c.Age, 
                c.Region,
                c.Town,
                c.Occupation, 
                c.Product,
            Price + AdditionalFee as TotalContractValue,     
        FROM `afcproj.files_dupe.Contracts_20201117` c
        join `afcproj.files_dupe.dec_17_cohort` j
            on c.ContractId = j.ContractId
            """
    cfdf = pd.read_gbq(contract_sql, index_col='ContractId', dialect='standard')  #.astype('float64')
    
    all_features = pd.merge(total_paid,
             cfdf,
             how='inner',
             left_index=True,
             right_index=True).sort_index()
    
    
    all_features = pd.merge(all_features,
             six_month_mark,
             how='inner',
             left_index=True,
             right_index=True,
             suffixes=['_total','_6m']).sort_index()
    
    
    
    all_features['percent_paid'] = all_features['AmountPaid_total']/all_features['TotalContractValue']
    fig, ax = plt.subplots()
    plt.scatter(all_features['days_dropped_6m'], all_features['percent_paid'])
    plt.xlabel('Total Number of Days Without Electricity at 6 Months')
    plt.ylabel('Total Percentage of Contract Value Paid at 6 Months')
    plt.savefig('files\\scatter days dropped vs amount paid')
    
    fig, ax = plt.subplots()
    plt.scatter(all_features['AmountPaid_6m'], all_features['percent_paid'])


# ## smaller df - these guys are interesting because they both pay back the same but one is regular payer and the other is bulk
# small_daily_sdf = daily_sdf.loc[['1574640',  '1574676']]

# pivoted = small_daily_sdf[['days_dropped', 'AmountPaid']].unstack('ContractId').fillna(0).sort_index().cumsum(axis=0)

# pivoted.plot(secondary_y=[('days_dropped', '1574640'),
#             ('days_dropped', '1574676'),], legend=False) 

