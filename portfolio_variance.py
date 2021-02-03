# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:42:03 2021

@author: mark
"""

import pandas as pd


SQL = """ 
select sum(AmountPaid) as SumAmountPaid, 
       sum(AmountPaid)/sum(TotalContractValue) as PercSumAmountPaid, 
        EXTRACT(YEAR FROM TransactionTS) AS year,
        EXTRACT(MONTH FROM TransactionTS) AS month,
        EXTRACT(DAY FROM TransactionTS) AS day
from
    (
     Select p.TransactionTS,
        p.AmountPaid,
        p.ContractId, 
        c.RegistrationDate, 
        p.Duration, 
        c.Price + c.AdditionalFee as TotalContractValue,     
        EXTRACT(YEAR FROM c.RegistrationDate) AS cohort_year,
        EXTRACT(MONTH FROM c.RegistrationDate) AS cohort_month,

    FROM afcproj.files_dupe.Payments_2020_11_17 p
    inner join afcproj.files_dupe.Contracts_20201117 c
        on c.ContractId = p.ContractId
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
        0 as Duration,
        c.Price + c.AdditionalFee as TotalContractValue,     
        EXTRACT(YEAR FROM c.RegistrationDate) AS cohort_year,
        EXTRACT(MONTH FROM c.RegistrationDate) AS cohort_month,

    FROM afcproj.files_dupe.Adjustments_2020_11_17 a
    inner join afcproj.files_dupe.Contracts_20201117 c
        on c.ContractId = a.ContractId
    WHERE a.BalanceChangeType = 'MANUAL'
        and (c.Product = 'X850'
        or c.Product = 'X850 Plus')
    )
    group by 
    year,
    month,
    day
    """

df = pd.read_gbq(SQL,)
df.index = pd.to_datetime(df[['year', 'month', 'day']])
df.sort_index(inplace=True)
df2 = df[['SumAmountPaid', 'PercSumAmountPaid']].astype('float64')

## June 2017 is when the data is relevant
df2 = df2.loc['Jun-2017':]
df2['PercSumAmountPaid'].plot()


## Daily standard deviation
df2.loc['Jan-2018':].std()


## Monthly standard deviation
df2.loc['Jan-2018':].groupby(pd.Grouper(freq='M')).sum().std()