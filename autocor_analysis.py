# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:27:33 2021

@author: mark
"""


#monthly_percent_pivot.iloc[:, 500:750].plot(legend=False)

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from matplotlib import pyplot as plt

from basic_datasets import MONTHLY_UNFINISHED_CONTRACTS, MONTHLY_SDF_PIVOT, BasicDatasets

bd = BasicDatasets()
unfinished_contracts = MONTHLY_UNFINISHED_CONTRACTS
monthly_sdf_pivot = MONTHLY_SDF_PIVOT

dfts = bd.daily_ts #full dataset excluding paid off contracts
monthly_days_of_elec = (~dfts['elec_is_off']).groupby(['ContractId', pd.Grouper(freq='1M', level=1)]).sum()

mde_pivot = monthly_days_of_elec.unstack(0)

acf_dict1 = {}
acf_dict2 = {}
acf_dict3 = {}
acf_dict4 = {}

for cprty in unfinished_contracts.columns:
    s1 = unfinished_contracts.loc[:, cprty]
    s2 = monthly_sdf_pivot.loc[:, cprty]
    s3 = mde_pivot.loc[:, cprty]
    #plot_acf(s)

    acf_dict1[cprty] = pd.Series(sm.tsa.acf(s1, nlags=12, missing='drop'))
    acf_dict2[cprty] = pd.Series(sm.tsa.acf(s2, nlags=12, missing='drop'))
    acf_dict3[cprty] = pd.Series(sm.tsa.acf(s3, nlags=12, missing='drop'))
    acf_dict4[cprty] = pd.Series(sm.tsa.acf(s3, nlags=12, missing='none'))
    
df1 = pd.DataFrame.from_dict(acf_dict1)
df2 = pd.DataFrame.from_dict(acf_dict2)
df3 = pd.DataFrame.from_dict(acf_dict3)
df4 = pd.DataFrame.from_dict(acf_dict4)

#print(df1.mean(axis=1))
#print(df2.mean(axis=1))
print(df3.mean(axis=1))
print(df3.mean(axis=0))
print(df4.mean(axis=1))


df3.to_csv('temp.csv')

best_params = df1.mean(axis=1).loc[1:6]

"""
TO LOOK INTO 
1) IS IT "DEFAULT" OR MIGRATION THAT CAUSES LOSSES
    2) PORTFOLIO AND COHORT DAILY/MONTHLY VARIANCE
    3) SEGMENT DEFAULT RATE TIME SERIES - par30, NEVER_PAY_AGAIN # LOOK INTO NEVERPAYAGAIN
        P&L IMPACT OF DEFAULTS
        
        4) MODEL FOR PREDICTING "FINAL VALUE" AT 36 MONTHS, OR EXPECTED REPAYMENT DATE
    
    LOOK INTO VASICEK
"""