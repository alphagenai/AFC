# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:48:47 2020

@author: mark
"""

import pandas as pd
import datetime as dt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from individual_analysis1 import small_df, cumulative_percent_sdf



## assemble features

if 0:
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
             right_index=True).sort_index()


#### Super Simple Regression Model
#currently only works for one contract at a time
df = cumulative_percent_sdf[cumulative_percent_sdf.columns[0:1]] 
diff_df = df.diff()
monthly = df.groupby(pd.Grouper(freq='M')).sum().cumsum().stack().reset_index()


#enc = OneHotEncoder(handle_unknown='ignore')
#enc.fit(monthly['ContractId'].to_numpy().reshape(-1,1))

ols = LinearRegression()
X = monthly.index.map(dt.datetime.toordinal).to_numpy().reshape(-1,1)
y = monthly.to_numpy()
ols.fit(X, y)
y_pred = ols.predict(X)

# Plot
plt.scatter(X, y,  color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)



