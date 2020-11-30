# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:47:04 2020

@author: mark
"""


import os
import pandas as pd
from google.cloud import bigquery
#from google.oauth2 import service_account


key_path = r"C:\Users\mat4m_000\Documents\Wellow data\SFC\AFCproj-keyfile.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

client = bigquery.Client()


SQL = """ 
    Select p.TransactionTS,
        p.AmountPaid,
        p.ContractId
    FROM afcproj.files_dupe.Payments_2020_11_17 p
    inner join afcproj.files_dupe.jan_19_cohort j
    on p.ContractId = j.ContractId    
        union
    Select a.TransactionTS,
        a.Amount,
        a.ContractId
    FROM afcproj.files_dupe.Adjustments_2020_11_17 a
    inner join afcproj.files_dupe.jan_19_cohort j
        on a.ContractId = j.ContractId    
        
            """

df = pd.read_gbq(SQL)

