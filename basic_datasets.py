# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:22:53 2021

@author: mark
"""

import pandas as pd

from individual_analysis1 import create_percent_sdf
from calculate_days_dropped import calculate_days_dropped

"""
TO DO: make all(/relevant) datasets exclude fully paid contracts
"""


def create_percent_sdf(input_df, cumulative=True, use_monthdiff=False, cohort='jan_19'):
    ### Get Contract info
    
    SQL = """
        SELECT c.ContractId,
            Price + AdditionalFee as TotalContractValue,     
            --c.RegistrationDate 
        FROM `afcproj.files_dupe.Contracts_20201117` c
        join `afcproj.files_dupe.{}_cohort` j
            on c.ContractId = j.ContractId
        """.format(cohort)
    #cdf = pd.read_gbq(SQL,index_col='ContractId').astype('float64')
    #cdf.to_pickle('files\\contract_df_{}.pkl'.format(cohort))
    cdf = pd.read_pickle('files\\contract_df_{}.pkl'.format(cohort))
    
    contract_ts = pd.merge(
        input_df.T,
        cdf,
        how='inner',
        left_index=True,
        right_index=True)
    
    contract_values = contract_ts['TotalContractValue']
    
    return_df = contract_ts.divide(contract_values, axis=0).drop(columns=['TotalContractValue']).T
    if cumulative:
        return_df = return_df.cumsum(axis=0)
    if use_monthdiff:
        pass
    else:
        return_df.index = pd.to_datetime(return_df.index,format='%Y/%m/%d %H:%M:%S')
        return_df.index.rename('TransactionTS', inplace=True)
    return return_df



class BasicDatasets(object):
    def __init__(self, cohort='dec_17'):
        self.cohort = cohort
        df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
        df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
        self.df = df.groupby(['ContractId', 'TransactionTS']).sum()
        
    @property
    def monthly_sdf(self):
        return self.df.groupby(['ContractId', pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
    
    @property
    def monthly_sdf_pivot(self):
        return self.monthly_sdf.unstack(0).fillna(0)

    @property
    def monthly_percent_pivot(self):
        return create_percent_sdf(self.monthly_sdf_pivot, cumulative=False, cohort=self.cohort)


    @property
    def monthly_cumulative_percent_sdf_pivot(self):
        try:
            return self._monthly_cumulative_percent_sdf_pivot
        except AttributeError:
            self._monthly_cumulative_percent_sdf_pivot = create_percent_sdf(self.monthly_sdf_pivot, cumulative=True, cohort=self.cohort)
            return self._monthly_cumulative_percent_sdf_pivot



    @property
    def daily_sample_payment_series(self):
        """ Series just containing payment info """
        return self.df.groupby(['ContractId', pd.Grouper(freq='D', level=1)])['AmountPaid'].sum()


    @property
    def daily_sdf(self):
        """ Dataframe containing all data for the 1,000 sample """
        return self.df.groupby(['ContractId', pd.Grouper(freq='D', level=1)]).sum()
    
    @property
    def daily_sdf_pivot(self):
        return self.daily_sample_payment_series.unstack(0).fillna(0)

    @property
    def daily_percent_pivot(self):
        return create_percent_sdf(self.daily_sdf_pivot, cumulative=False, cohort=self.cohort)

    @property
    def daily_cumulative_percent_sdf_pivot(self): 
        return create_percent_sdf(self.daily_sdf_pivot, cumulative=True, cohort=cohort)

    @property
    def daily_full_ts(self):
        """ includes electricity used and PAR30+ """
        return calculate_days_dropped(self.df.groupby(['ContractId', pd.Grouper(freq='D', level=1)]).sum())
    
    @property
    def daily_fully_paid(self):
        return self.daily_cumulative_percent_sdf_pivot.shift(1) >= 0.99

    @property
    def daily_ts(self):
        """ full ts excluding completed contracts """
        full_df = pd.merge(self.daily_full_ts,
                        self.daily_cumulative_percent_sdf_pivot.stack().swaplevel().to_frame('cum_val'),
                        left_index=True,
                        right_index=True)
        return full_df[full_df['cum_val'] < 0.999]
    


cohort='dec_17'
bds = BasicDatasets()
df = bds.df
MONTHLY_SDF = monthly_sdf = df.groupby(['ContractId', pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
MONTHLY_SDF_PIVOT = monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
MONTHLY_PERCENT_PIVOT = monthly_percent_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=False, cohort=cohort)

DAILY_SDF = daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='D', level=1)])['AmountPaid'].sum()
DAILY_SDF_PIVOT = daily_sdf_pivot = daily_sdf.unstack(0).fillna(0)

DAILY_CUMULATIVE_PERCENT_SDF_PIVOT = daily_cumulative_percent_sdf_pivot = create_percent_sdf(daily_sdf_pivot, cumulative=True, cohort=cohort)
MONTHLY_CUMULATIVE_PERCENT_SDF_PIVOT = monthly_cumulative_percent_sdf_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=True, cohort=cohort)

DAILY_FULLY_PAID = daily_fully_paid = daily_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag
MONTHLY_FULLY_PAID = monthly_fully_paid = monthly_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag


MONTHLY_UNFINISHED_CONTRACTS = monthly_percent_pivot.mask(monthly_fully_paid)

if __name__ == "__main__":
        
    cohort='dec_17'
    bd = BasicDatasets()
    df = bd.df
    dts = bd.daily_full_ts
    
    full_ts_with_cum_val = pd.merge(dts,
        bd.daily_cumulative_percent_sdf_pivot.stack().swaplevel().to_frame('cum_val'),
        left_index=True,
        right_index=True)
    
    cid = '1349704'

    df.loc[cid]
    
    dts.loc[cid].loc['Sept-2020']
    
    full_ts_with_cum_val.loc[cid].loc['Sept-2020']