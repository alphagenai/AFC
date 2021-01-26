# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:43:25 2021

@author: mark
"""

import pandas as pd

from individual_analysis1 import month_diff


"""
TO DO: USE THE PERCENTAGE CHANGE IN PERCENTAGE REPAYMENT FROM THE PREVIOUS MONTH COHORT!

Why am i reinventing the wheel here? isnt this already implemented in moving_average_model?

"""







""" SIMPLE MOVING AVERAGE-BASED FORECAST """

## just removed the word _cohort
PAYMENT_DATA_SQL = """ 
    Select p.TransactionTS,
        p.AmountPaid,
        p.ContractId, 
        c.RegistrationDate, 
        p.Duration
    FROM afcproj.files_dupe.Payments_2020_11_17 p
    inner join afcproj.files_dupe.{0} j
        on p.ContractId = j.ContractId   
    inner join afcproj.files_dupe.Contracts_20201117 c
        on c.ContractId = j.ContractId
    WHERE p.paymentStatusTypeEntity != 'REFUSED'
        and
        p.PaymentResultTypeEntity != 'PAYMENT_FREE_CREDIT'
        and (c.Product = 'X850'
        or c.Product = 'X850 Plus')
    UNION ALL
    Select a.CreatedAt,
        a.Amount,
        a.ContractId, 
        c.RegistrationDate, 
        0 as Duration
    FROM afcproj.files_dupe.Adjustments_2020_11_17 a
    inner join afcproj.files_dupe.{0} j
        on a.ContractId = j.ContractId
    inner join afcproj.files_dupe.Contracts_20201117 c
        on c.ContractId = j.ContractId

    WHERE a.BalanceChangeType = 'MANUAL'
        and (c.Product = 'X850'
        or c.Product = 'X850 Plus')
        """

cohort = 'September_2019'

forecast_startdate = '2019-6-30'
forecast_monthdiff = 18


sql = PAYMENT_DATA_SQL.format(cohort)
            
df = pd.read_gbq(sql,)
df = df.set_index(['ContractId'])
df['monthdiff'] = month_diff(df['TransactionTS'].dt.tz_localize(None), df['RegistrationDate']).clip(0,None)
df = df.groupby(['ContractId','monthdiff']).agg({
    'TransactionTS':'count', 
    'AmountPaid': 'sum',
    'Duration':'sum',
    })    

df = df['AmountPaid'].astype('float64').sort_index().to_frame()

df['paid_cumsum'] = df.groupby(['ContractId',]).cumsum()

contract_SQL = """
    SELECT c.ContractId,
        Price + AdditionalFee as TotalContractValue,     
        --c.RegistrationDate 
    FROM `afcproj.files_dupe.Contracts_20201117` c
    join `afcproj.files_dupe.{}` j
        on c.ContractId = j.ContractId
    """.format(cohort)
cdf = pd.read_gbq(contract_SQL,index_col='ContractId').astype('float64')


contract_ts = df.join(cdf, how='inner')
contract_values = contract_ts['TotalContractValue']


contract_ts['percent_paid'] = contract_ts['AmountPaid'] / contract_ts['TotalContractValue']

contract_ts['moving_average_6m'] = contract_ts.reset_index('ContractId').groupby(
    'ContractId')['percent_paid'].rolling(window=6).mean()

contract_ts['moving_average_6m_cumsum'] = contract_ts.groupby('ContractId')['moving_average_6m'].cumsum()

contract_ts['perc_moving_average_6m_cumsum'] = contract_ts['moving_average_6m_cumsum'] / contract_ts['TotalContractValue']

contract_ts['simple_6m_forecast'] = pd.concat(contract_ts['percent_paid'].loc[:, :forecast_monthdiff]
                                              
                                            
contract_ts['simple_cum_6m_forecast'] = contract_ts['simple_6m_forecast'].groupby .apply(lambda x:np.min([x, 1.0]))