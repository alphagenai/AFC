# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 00:12:12 2020

@author: mark
"""



default = daily_sdf_fullts['PAR30+']
default_event = (default.diff() != 0) & (daily_sdf_fullts['PAR30+']==True)
recovery_event = (default.diff() != 0) & (daily_sdf_fullts['PAR30+']==False) & (
    (default.index.get_level_values(1).month != 12) & (default.index.get_level_values(1).year != 2017)
    )
daily_sdf_fullts[recovery_event].to_csv('recoveries.csv')
daily_sdf_fullts[default_event].to_csv('defauls.csv')



#############  FREQUENTIST


## 1) MONTH OF DEFAULT DOES NOT MATTER
monthly_sdf = daily_sdf.groupby(['ContractId',pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)

first_payment = monthly_sdf_pivot.iloc[1]

first_payment[first_payment==0]

defaults = monthly_sdf_pivot.iloc[1:]==0
individual_uncond_PD = defaults.sum()/defaults.count()
unconditional_PD = defaults.sum().sum()/defaults.count().sum()

last_month_defaults = defaults.shift(1)

PD_given_D = defaults & last_month_defaults
PD_given_ND = defaults & ~last_month_defaults.astype('bool')

PND_given_D = ~defaults.astype('bool') & last_month_defaults
PND_given_ND = ~defaults.astype('bool') & ~last_month_defaults.astype('bool')



# monthly_sdf_pivot.to_csv('monthly_df.csv')
# defaults.to_csv('defaults.csv')
# PD_given_D.to_csv('PD_given_D.csv')
# PD_given_ND.to_csv('PD_given_ND.csv')



## 2) MONTH MATTERS