# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:10:37 2020

@author: mark
"""

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from individual_analysis1 import create_small_df, create_percent_sdf, convert_to_daily_pivot


## This guy is interesting:
# cumulative_percent_sdf['1351574'].plot()



def pick_interesting_timepoints(cumulative_df):
    at_month_6 = cumulative_df.loc['2018-05-30']
    at_month_18 = cumulative_df.loc['2019-05-31']
    at_month_24 = cumulative_df.loc['2019-11-30']
    last_chance = cumulative_df.loc['2020-11-17']
    return (at_month_6, at_month_18, at_month_24, last_chance)


def df_autocorr(df, lag=1, axis=0):
    """Compute full-sample column-wise autocorrelation for a DataFrame."""
    return df.apply(lambda col: col.autocorr(lag), axis=axis)


def payment_histograms():
    fig, ax = plt.subplots()
    sns.histplot(at_month_6, bins=100)
    title = '6 month payment histogram'
    plt.title(title)
    plt.savefig(title)
    
    fig, ax = plt.subplots()
    sns.histplot(at_month_18, bins=100)
    title = '18 month payment histogram'
    plt.title(title)
    plt.savefig(title)
    
    fig, ax = plt.subplots()
    sns.histplot(at_month_24, bins=100)
    title = '24 month payment histogram'
    plt.title(title)
    plt.savefig(title)
    
    fig, ax = plt.subplots()
    sns.histplot(last_chance, bins=100)
    title = '3 year payment histogram'
    plt.title(title)
    plt.savefig(title)
        
    plt.show()
    
def create_boolean_categories(cumulative_percent_sdf):
    at_month_6, at_month_18, at_month_24, last_chance = pick_interesting_timepoints(cumulative_percent_sdf)
    paid_off_on_time = (at_month_18 >= 0.98)
    paid_off_eventually = (at_month_18 < 0.98) & (last_chance >= 0.98)
    gt_80 = (last_chance > 0.8) & (last_chance < 0.980)
    bad = (last_chance < 0.8) & (last_chance >= 0.5)
    very_bad = (last_chance < 0.5) & (last_chance >= 0.2) 
    default = (last_chance < 0.2)  
    return paid_off_on_time, paid_off_eventually, gt_80, bad, very_bad, default


if __name__ == "__main__":
    
    #small_df = create_small_df(size=1000, cohort='dec_17')
    small_df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    daily_sdf_pivot = convert_to_daily_pivot(small_df)
    cumulative_percent_sdf = create_percent_sdf(daily_sdf_pivot, cumulative=True, cohort='dec_17')
    cumulative_amount_sdf = daily_sdf_pivot.cumsum(axis=0)
    cumulative_amount_sdf.index = pd.to_datetime(cumulative_amount_sdf.index,format='%Y/%m/%d %H:%M:%S')
    


    paid_off_on_time, paid_off_eventually, gt_80, bad, very_bad, default = create_boolean_categories(cumulative_percent_sdf)
    categories = {'Paid off on time':paid_off_on_time,
              'Paid off after 18 months':paid_off_eventually ,
              'Almost fully repaid':gt_80 ,
              'Less than 80% paid':bad ,
              'Less than 50% paid':very_bad ,
              'Less than 20% paid':default,

              }

    run_sum = 0
    for cat_name, cat in categories.items():
        print(cat_name, cat.sum())
        run_sum += cat.sum()
    ## should add to 1000!
    print(run_sum)
    

    
    cumulative_percent_sdf[paid_off_on_time.index[paid_off_on_time]].plot()
    
    
    percent_sdf = create_percent_sdf(daily_sdf_pivot, cumulative=False, cohort='dec_17')
    
    at_month_6, at_month_18, at_month_24, last_chance = pick_interesting_timepoints(cumulative_percent_sdf)
    
    
    
    fig, ax = plt.subplots()
    sns.histplot(last_chance[gt_80 | bad | very_bad | default], bins=100)
    plt.show()
    
    
    
    ## look at individual and average variance
    mean_daily_repayment = {}
    std_daily_repayment = {}
    std_cust_repayment = {}
    full_cumulative_repayment = {}
    
    
    
    for cat_name, cat in categories.items():
        mean_daily_repayment[cat_name] = percent_sdf[cat.index[cat]].mean(axis=1)
        std_daily_repayment[cat_name] = percent_sdf[cat.index[cat]].std(axis=1)
        std_cust_repayment[cat_name] = percent_sdf[cat.index[cat]].std(axis=0)
        full_cumulative_repayment[cat_name] = cumulative_percent_sdf[cat.index[cat]]
        
        # daily average payments across the group
        fig, ax = plt.subplots()
        mean_daily_repayment[cat_name].plot(title=cat_name, legend=False)
        plt.savefig('files\\{}_daily_average_payments'.format(cat_name))
        ## look into changing the scale to max 0.002

        # daily variance across the group
        fig, ax = plt.subplots()
        std_daily_repayment[cat_name].plot(title=cat_name, legend=False)
        plt.savefig('files\\{}_daily_variance_payments'.format(cat_name))
        
        
        ## what do the cumulative payments look like?
        fig, ax = plt.subplots()
        full_cumulative_repayment[cat_name].plot(title=cat_name, legend=False)
        plt.savefig('files\\{}_cumulative_payments'.format(cat_name))
        
        ##histograms of daily payments
        # these are not very useful because most days are zero - might be more 
        # interesting to look at this on the monthly level
        fig, ax = plt.subplots()
        sns.histplot(percent_sdf[cat.index[cat]], bins=100, legend=False)
        plt.xscale('log')
        title = 'Histogram of Daily Payments {}'.format(cat_name)
        plt.title(title)
        plt.savefig('files\\'+title)
    plt.show() 
    
    
    for cat_name, cat in categories.items():
    
        ## histogram at 6 months
        fig, ax = plt.subplots()
        sns.histplot(last_chance[cat.index[cat]], bins=20, legend=False)
        title = 'Histogram of Cumulative Payments at 3 Years {}'.format(cat_name)
        plt.title(title)
        plt.savefig('files\\'+title)
        
        ##what are the total number of token bought for each group?
        
    plt.show()
    
    
    
    
    ## cumulative % payment HISTOGRAM AT 18,24,36 Months
    
    fig, ax = plt.subplots()
    sns.histplot(at_month_6)
    plt.title('6 months')
    
    fig, ax = plt.subplots()
    sns.histplot(at_month_18)
    plt.title('18 months')
    
    fig, ax = plt.subplots()
    sns.histplot(at_month_24)
    plt.title('24 months')
    
    fig, ax = plt.subplots()
    sns.histplot(last_chance)
    plt.title('3 Years')
    
    
    ### 
    #AUTOCORRELATION
    ###
    
    
    monthly= percent_sdf.groupby(pd.Grouper(freq='M')).sum()
    df_autocorr(monthly, lag=1, axis=0).plot()
    autocorr = df_autocorr(monthly, lag=1, axis=0)
    autocorr.sort_values().plot()