# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:30:40 2021

@author: mark
"""


##loss analysis

import pandas as pd

from individual_analysis1 import create_percent_sdf


df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)

monthly_sdf = df.groupby(['ContractId',pd.Grouper(key='TransactionTS', freq='M',)])['AmountPaid'].sum()

monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)


monthly_cumulative_percent_sdf_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=True, cohort='dec_17')
        
defaults = (monthly_sdf_pivot==0).astype('boolean') # total non-NaN: 36,000 incl Dec; 28,593 incl. Jan 18
paid  = (monthly_sdf_pivot!=0).astype('boolean')
fully_paid = monthly_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag
        

SQL = """
    SELECT c.ContractId,
        Price + AdditionalFee as TotalContractValue,     
        --c.RegistrationDate 
    FROM `afcproj.files_dupe.Contracts_20201117` c
    join `afcproj.files_dupe.{}_cohort` j
        on c.ContractId = j.ContractId
    """.format("dec_17")
cdf = pd.read_gbq(SQL,index_col='ContractId').astype('float64')

contract_ts = pd.merge(
    monthly_sdf_pivot.T,
    cdf,
    how='inner',
    left_index=True,
    right_index=True)

contract_values = contract_ts['TotalContractValue']

deposit = contract_values - contract_ts[contract_ts.columns[0]]
contractual_amount = (contract_values - deposit)/18.  # in AFC's spreadsheet they use approx. 19 months as contract term
deficit = monthly_sdf_pivot - contractual_amount 
cum_deficit = 
default_losses = deficit == -contractual_amount 

deficit[default_losses].mask(fully_paid)
deficit[~default_losses].mask(fully_paid)

monthly_sdf_pivot.mask(fully_paid)

"""
monthly_payments = monthly_sdf_pivot.sum(axis=1).to_frame('total_paid')

monthly_payments['contractual'] = 0
monthly_payments['contractual'].loc['jan-2018':'jun-2019'] =  (contract_values.sum() - monthly_payments['total_paid'].values[0]) /18.

monthly_payments.plot(kind='bar')

monthly_payments['max_electricity'] = monthly_payments.index.daysinmonth*55*1000

monthly_payments['deficit'] = monthly_payments['max_electricity'] - monthly_payments['total_paid'] 
"""