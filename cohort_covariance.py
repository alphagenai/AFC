# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:20:19 2021

@author: mark
"""

import pandas as pd

from individual_analysis1 import month_diff


using_monthdiff = False


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

date_range_start = '2017-06-01'
date_range_end = '2020-10-31'

drange = pd.date_range(date_range_start, date_range_end, freq='1MS')

cohort_dict = {}

for startmonth, endmonth in zip(drange, drange.shift(1)):
    cohort_name = '{}_{}'.format(startmonth.month_name(),startmonth.year,)
    cutoff_month = endmonth.month_name()
    
    
    sql = PAYMENT_DATA_SQL.format(cohort_name)
                
    df = pd.read_gbq(sql,)
    df = df.set_index(['ContractId'])
    
    
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    
    df = df.astype('float64', errors='ignore')
    df2 = df.groupby([pd.Grouper(key='TransactionTS', freq='D')]).sum().sort_index()
    #df2.loc['Oct-2019':, 'AmountPaid'].plot()
    

    if using_monthdiff:
        df['monthdiff'] = month_diff(df['TransactionTS'].dt.tz_localize(None), df['RegistrationDate']).clip(0,None)
        df = df.groupby(['ContractId','monthdiff']).agg({
            'TransactionTS':'count', 
            'AmountPaid': 'sum',
            'Duration':'sum',
            })    
        
        df = df['AmountPaid'].astype('float64').sort_index().to_frame()

    cohort_dict[cohort_name] = df2.loc[:, 'AmountPaid']

    cohort_df = pd.DataFrame.from_dict(cohort_dict)
    cohort_df.to_pickle('cohort_payments.pkl')
