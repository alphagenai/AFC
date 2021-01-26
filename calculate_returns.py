# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:38:04 2021

@author: mark
"""



df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
df = df.groupby(['ContractId', 'TransactionTS']).sum()
daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
daily_sdf_pivot = daily_sdf['AmountPaid'].unstack(0).fillna(0).sort_index()

daily_cumulative_percent_sdf_pivot = create_percent_sdf(daily_sdf_pivot, 
                                                          cumulative=True, 
                                                          
                                                          
cohort='dec_17'
input_df = daily_sdf_pivot
SQL = """
     SELECT c.ContractId,
         Price + AdditionalFee as TotalContractValue,     
         --c.RegistrationDate 
     FROM `afcproj.files_dupe.Contracts_20201117` c
     join `afcproj.files_dupe.{}_cohort` j
         on c.ContractId = j.ContractId
     """.format(cohort)
cdf = pd.read_gbq(SQL,index_col='ContractId').astype('float64')
 
 
contract_ts = pd.merge(
     input_df.T,
     cdf,
     how='inner',
     left_index=True,
     right_index=True)
 



"""
Can we bck out Rsq from par30 flags?
"""