# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:23:07 2020

@author: mark
"""

import pandas as pd
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
    daily_sdf['days_dropped'] = (daily_sdf['TransactionTS']  - daily_sdf['prev_payment_date'] - pd.to_timedelta(daily_sdf['prev_duration'], unit='D')).dt.days

    
    ## stupid americans
    date_idx = pd.date_range('12/01/2017', '30-11-2020')
    idx = pd.MultiIndex.from_product(
        [daily_sdf.index.levels[0], date_idx], names=['ContractId', 'TransactionTS'])

    daily_sdf_fullts = daily_sdf.reindex(idx,).fillna({'Duration':-1, 
                                                       'AmountPaid':0})
#    daily_sdf_fullts['days_elec_left'] = daily_sdf_fullts.groupby(['ContractId', 'Duration']).cumsum(axis=0)    #['Duration'].cumsum(axis=0).apply(lambda x : np.max([0,x]))
#    daily_sdf_fullts['days_without'] = daily_sdf_fullts.groupby(level=0)['Duration'].apply(lambda x: 0 if x > 0 else x)


#    daily_sdf_fullts['days_elec_left'] = daily_sdf_fullts.loc[daily_sdf_fullts['Duration'] > 0, ['Duration']].cumsum(axis=0)
    daily_sdf_fullts['cumsum'] = daily_sdf_fullts.groupby(level=0)['Duration'].cumsum(axis=0)
    daily_sdf_fullts['nan_non_zero'] = daily_sdf_fullts['cumsum'].mask(daily_sdf_fullts['Duration'] > 0)
    daily_sdf_fullts['ffill'] = daily_sdf_fullts['nan_non_zero'].ffill()

    daily_sdf_fullts['final'] = daily_sdf_fullts['cumsum'].sub(daily_sdf_fullts['nan_non_zero'].ffill(), fill_value=0)
    daily_sdf_fullts.to_csv('temp.csv')
    return daily_sdf




if __name__ == "__main__":

    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)

    df = df.groupby(['ContractId', 'TransactionTS']).sum()

    daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()


    daily_sdf = calculate_days_dropped(daily_sdf)


# ## smaller df - these guys are interesting because they both pay back the same but one is regular payer and the other is bulk
# small_daily_sdf = daily_sdf.loc[['1574640',  '1574676']]

# pivoted = small_daily_sdf[['days_dropped', 'AmountPaid']].unstack('ContractId').fillna(0).sort_index().cumsum(axis=0)

# pivoted.plot(secondary_y=[('days_dropped', '1574640'),
#             ('days_dropped', '1574676'),], legend=False) 



    
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
