# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:36:25 2020

@author: mark
"""

SQL = """
    SELECT sum(p.AmountPaid) as total_amount_paid,
        sum(DISTINCT c.ContractId)
        EXTRACT(YEAR FROM RegistrationDate) AS cohort_year,
        EXTRACT(MONTH FROM RegistrationDate) AS cohort_month,
    FROM `afcproj.files_dupe.Payments_2020_11_17` p
    join `afcproj.files_dupe.Contracts_20201117`  c
        on p.ContractId = p.ContractId
    WHERE p.paymentStatusTypeEntity != 'REFUSED'
        and
        p.PaymentResultTypeEntity != 'PAYMENT_FREE_CREDIT'
        and c.PaymentMethod = 'FINANCED'

    group by cohort_month, cohort_year

    """
    
SQL = """
    SELECT count(c.ContractId) as total_amount_paid,
        
        EXTRACT(YEAR FROM RegistrationDate) AS cohort_year,
        EXTRACT(MONTH FROM RegistrationDate) AS cohort_month,
    FROM `afcproj.files_dupe.Contracts_20201117`  c
    WHERE c.PaymentMethod = 'FINANCED'
    group by cohort_month, cohort_year, ContractId

    """
    
