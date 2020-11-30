# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:36:25 2020

@author: mark
"""

from afc_cohort_timeseries import make_cohort_columns_as_index

SQL = """  -- takes a very long time to run
    SELECT sum(p.AmountPaid) as total_amount_repaid,
        count(DISTINCT c.ContractId) as number_of_dist_contracts,
        EXTRACT(YEAR FROM RegistrationDate) AS cohort_year,
        EXTRACT(MONTH FROM RegistrationDate) AS cohort_month,
    FROM `afcproj.files_dupe.Payments_2020_11_17` p
    join `afcproj.files_dupe.Contracts_20201117`  c
        on p.ContractId = c.ContractId
    WHERE p.paymentStatusTypeEntity != 'REFUSED'
        and
        p.PaymentResultTypeEntity != 'PAYMENT_FREE_CREDIT'
        and c.PaymentMethod = 'FINANCED'

    group by cohort_month, cohort_year

    """
cohort_total_repaid = pd.read_gbq(SQL,)
    
SQL = """
     SELECT count(c.ContractId) as number_of_contracts,
         --count(DISTINCT c.ContractId) as dist_contracts
        SUM(Deposit) + SUM(Price) + SUM(AdditionalFee) as TotalContractValue, 
        
        EXTRACT(YEAR FROM RegistrationDate) AS cohort_year,
        EXTRACT(MONTH FROM RegistrationDate) AS cohort_month,
    FROM `afcproj.files_dupe.Contracts_20201117`  c
    WHERE c.PaymentMethod = 'FINANCED'
    group by cohort_month, cohort_year
    """
    
cohort_contract_value = pd.read_gbq(SQL,)

df = pd.merge(
    cohort_total_repaid,
    cohort_contract_value,
    how='outer',
    left_on=['cohort_year', 'cohort_month'],
    right_on=['cohort_year', 'cohort_month'],
    )

df = make_cohort_columns_as_index(df)

df['mean_value_per_contract'] = df['TotalContractValue']/df['number_of_contracts']
df['mean_repay_per_contract'] = df['total_amount_repaid']/df['number_of_contracts']
df['repay_per_contract_dollar'] = df['total_amount_repaid']/df['TotalContractValue']
