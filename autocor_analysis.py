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

acf_dict = {}

for col in monthly_percent_pivot.columns:
    s = monthly_percent_pivot.loc[:, col]
    #plot_acf(s)
    acf_dict[col] = pd.Series(sm.tsa.acf(s, nlags=12))
    
df = pd.DataFrame.from_dict(acf_dict)

df.mean(axis=1)


"""
TO LOOK INTO 
1) IS IT "DEFAULT" OR MIGRATION THAT CAUSES LOSSES
    2) PORTFOLIO AND COHORT DAILY/MONTHLY VARIANCE
    3) SEGMENT DEFAULT RATE TIME SERIES - par30, NEVER_PAY_AGAIN # LOOK INTO NEVERPAYAGAIN
        P&L IMPACT OF DEFAULTS
        
        4) MODEL FOR PREDICTING "FINAL VALUE" AT 36 MONTHS, OR EXPECTED REPAYMENT DATE
    
    LOOK INTO VASICEK
"""