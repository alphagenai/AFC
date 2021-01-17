# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:02:22 2021

@author: mark
"""


import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt

from individual_analysis1 import create_percent_sdf


df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)

df = df.groupby(['ContractId', 'TransactionTS']).sum()

daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
monthly_sdf = daily_sdf.groupby(['ContractId',pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)

def payment_correlations():
    
    mycorr = monthly_sdf_pivot.corr()
    mycorr.stack().describe()
    
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

def monthly_default_correlation():
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
        




""" 
'returns' fudged to be normally distributed payments 

"""


def latent_factor_correlations():
    monthly_percent_sdf_pivot = create_percent_sdf(monthly_sdf_pivot, 
                                                        cumulative=False, 
                                                        cohort='dec_17'
                                                        )
    
    
    ## might want to remove the first month - non payment before contract start and large payments
    monthly_cumulative_percent_sdf_pivot = create_percent_sdf(monthly_sdf_pivot, 
                                                              cumulative=True, cohort='dec_17')
    
    fully_paid = monthly_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag
        
    means = monthly_percent_sdf_pivot.mask(fully_paid).mean(axis=1)
    stds = monthly_percent_sdf_pivot.mask(fully_paid).std(axis=1)

    mean = monthly_percent_sdf_pivot.mask(fully_paid).stack().mean()
    std = monthly_percent_sdf_pivot.mask(fully_paid).stack().std()

    
    centered = monthly_percent_sdf_pivot.sub(mean)
    norm = centered.divide(std)
    
    
    pd.concat([means, stds,], axis=1).plot()
    
    
    mycorr = norm.corr()
    #mycov = norm.cov()
    #sns.heatmap(mycorr)
    mycorr.stack().describe()
    mycov.stack().describe()  # ???
    ## correlations look higher in 2020 but probably due to it being mainly bad payers left - need to investigate Dec 2019 cohort
    #annual_correlations(norm)
    fig, ax = plt.subplots()
    title = 'Monthly normalised latent factor against % payment'
    plt.scatter(norm.loc['2018':].stack(), monthly_percent_sdf_pivot.loc['2018':].stack(),)
    ax.set_xlim(-3, 6)
    ax.set_ylim(-0.025, 0.2)
    plt.title(title)
    plt.savefig('files\\'+title)

def pca_analysis():
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(norm)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

if __name__ == "__main__":
    pass