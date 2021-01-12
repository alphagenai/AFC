# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:02:22 2021

@author: mark
"""


import seaborn as sns

mycorr = monthly_sdf_pivot.corr()
sns.heatmap(mycorr)
mycorr > 0.8
(mycorr > 0.8).sum(axis=1)
high_corr = (mycorr > 0.8).sum(axis=1)
high_corr.sort_values()
highest_corr = high_corr.sort_values().tail(20)
monthly_sdf_pivot.loc[:,highest_corr.index]
monthly_sdf_pivot.loc[:,highest_corr.index].to_csv('high_correlation.csv')
high_corr_ts = monthly_sdf_pivot.loc[:,highest_corr.index]
high_corr_ts.cumsum(axis=0).plot()


### I THINK WHAT WE REALLY WANT IS STATE (aka default) CORRELATION

no_NAs = pd_calc._defaults.fillna(False)
mycorr = no_NAs.corr()
no_NAs.loc[:,mycorr.isna().all()]  # NAs are all false


## par30 correlation
par30_daily_pivot = daily_sdf_fullts['PAR30+'].unstack(0).sort_index()
mycorr= par30_daily_pivot.corr()
sns.heatmap(mycorr)

par30_daily_pivot.corr().stack().describe()

par30_daily_pivot.cov()
daily_sdf_pivot['2018'].corr().stack().describe()
daily_sdf_pivot['2019'].corr().stack().describe()
daily_sdf_pivot['2020'].corr().stack().describe()

def annual_correlations(df):
    for year in ['2018', '2019', '2020']:
        print(year+'\n')
        print(df[year].corr().stack().describe())
        
annual_correlations(monthly_sdf_pivot)




""" 
'returns' fudged to be normally distributed payments 

"""

monthly_percent_sdf_pivot = create_percent_sdf(monthly_sdf_pivot, 
                                                    cumulative=False, 
                                                    cohort='dec_17'
                                                    )


## might want to remove the first month - non payment before contract start and large payments

fully_paid = monthly_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag


means = monthly_percent_sdf_pivot.mask(fully_paid).mean(axis=1)
stds = monthly_percent_sdf_pivot.mask(fully_paid).std(axis=1)

centered = monthly_percent_sdf_pivot.sub(means, axis=0)
norm = centered.divide(stds, axis=0)


pd.concat([means, stds,], axis=1).plot()


mycorr = norm.corr()
sns.heatmap(mycorr)
mycorr.stack().describe()

