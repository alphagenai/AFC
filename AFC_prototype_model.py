# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:48:47 2020

@author: mark
"""

import pandas as pd

from individual_analysis1 import small_df

## assemble features

contract_sql = """
    SELECT c.*,
        Price + AdditionalFee as TotalContractValue,     
    FROM `afcproj.files_dupe.Contracts_20201117` c
    join `afcproj.files_dupe.jan_19_cohort` j
        on c.ContractId = j.ContractId
        """
cdf = pd.read_gbq(contract_sql,index_col='ContractId') #.astype('float64')


all_features = pd.merge(small_df,
         cdf,
         how='inner',
         left_index=True,
         right_index=True)
