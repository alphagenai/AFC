# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:47:04 2020

@author: mark
"""


import google_sa_auth
import pandas as pd
from google.cloud import bigquery
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib as mpl
import matplotlib.ticker as ticker
import time
import sqlite3

from sqlite_settings import DB_PATH

PAYMENT_DATA_SQL = """ 
    Select p.TransactionTS,
        p.AmountPaid,
        p.ContractId, 
        c.RegistrationDate, 
        p.Duration
    FROM afcproj.files_dupe.Payments_2020_11_17 p
    inner join afcproj.files_dupe.{0}_cohort j
        on p.ContractId = j.ContractId   
    inner join afcproj.files_dupe.Contracts_20201117 c
        on c.ContractId = j.ContractId
    WHERE p.paymentStatusTypeEntity != 'REFUSED'
        and
        p.PaymentResultTypeEntity != 'PAYMENT_FREE_CREDIT'
        and (c.Product = 'X850'
        or c.Product = 'X850 Plus')
    UNION ALL
    Select a.CreatedAt,
        a.Amount,
        a.ContractId, 
        c.RegistrationDate, 
        0 as Duration
    FROM afcproj.files_dupe.Adjustments_2020_11_17 a
    inner join afcproj.files_dupe.{0}_cohort j
        on a.ContractId = j.ContractId
    inner join afcproj.files_dupe.Contracts_20201117 c
        on c.ContractId = j.ContractId

    WHERE a.BalanceChangeType = 'MANUAL'
        and (c.Product = 'X850'
        or c.Product = 'X850 Plus')
        """

def month_diff(a, b):
    return 12 * (a.dt.year - b.dt.year) + (a.dt.month - b.dt.month)


def create_small_df(size=100, limit=False, use_monthdiff=False, random_seed=42, 
                    cohort='jan_19', use_bigquery=True):

    try:
        df = pd.read_pickle('files\\small_df_{}_{}.pkl'.format(size, cohort))
    except:
        sql = PAYMENT_DATA_SQL.format(cohort)
        if limit:
            sql = sql + " LIMIT {}".format(limit)
        if use_bigquery:
            print("Reading from bigquery")
            tic = time.perf_counter()
            df = pd.read_gbq(sql,) #chunksize=10000) #chunksize doesnt work
            toc = time.perf_counter()
            print(f"Time taken to read the database: {toc - tic:0.4f} seconds")
        else:
            print("Reading from sqlite")
            tic = time.perf_counter()
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_gbq(sql,con=conn)
            toc = time.perf_counter()
            print(f"Time taken to read the database: {toc - tic:0.4f} seconds")

        df = df.set_index(['ContractId'])
    
        df = reduce_df_size(df, size=size, random_seed=random_seed)
        df = df.astype('float64', errors='ignore')  ## datetime columns cause errors
    
        df.to_pickle('files\\small_df_{}_{}.pkl'.format(size, cohort))
    
    df['next_payment_date'] = df.shift(-1)['TransactionTS']
    
    if use_monthdiff:
        df['monthdiff'] = month_diff(df['TransactionTS'].dt.tz_localize(None), df['RegistrationDate']).clip(0,None)
        df = df.groupby(['ContractId','monthdiff']).agg({
            'TransactionTS':'count', 
            'AmountPaid': 'sum',
            'Duration':'sum',
            })    
    else:
        
        df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
        
        df = df.groupby(['ContractId','TransactionTS']).max() # by using max we are assuming all entries are unique
                  
    return df.sort_index()

def reduce_df_size(df, size, random_seed=42):
    random.seed(a=random_seed)        
    sample_random_IDs = random.sample(df.index.unique().tolist(), k=size,)
    
    small_df = df.loc[sample_random_IDs]   # see which IDs --> small_df.index.get_level_values(0).unique()
    #small_df = df.loc[pd.read_pickle('files\\1000 contractIds.pkl')]   # see which IDs --> small_df.index.get_level_values(0).unique()
    return small_df

def create_percent_sdf(input_df, cumulative=True, use_monthdiff=False, cohort='jan_19'):
    ### Get Contract info
    
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
    
    contract_values = contract_ts['TotalContractValue']
    
    return_df = contract_ts.divide(contract_values, axis=0).drop(columns=['TotalContractValue']).T
    if cumulative:
        return_df = return_df.cumsum(axis=0)
    if use_monthdiff:
        pass
    else:
        return_df.index = pd.to_datetime(return_df.index,format='%Y/%m/%d %H:%M:%S')
        return_df.index.rename('TransactionTS', inplace=True)
    return return_df

def convert_to_daily_pivot(small_df ):
    sdf_pivot = small_df['AmountPaid'].unstack(0).fillna(0).sort_index()
    
    daily_sdf_pivot = sdf_pivot.groupby(sdf_pivot.index.date).sum() #all payments in one day are grouped together
    return daily_sdf_pivot

def plot_daily_payments(df, batch_size=10, cohort_name='Jan 2019', y_formatter=None):
    num_pieces = int(len(df)/batch_size)
    for i in range(max(num_pieces,10)):
        title = "Cumulative Payments for {} Random X850 Contracts in {} Cohort".format(batch_size, cohort_name)
        columns = df.columns[i*batch_size:(i+1)*batch_size]
        ax = df[columns].plot(figsize=(20,8), legend=False, title=title)
        ax.set_xlabel("Transaction Date")
        ax.set_ylabel("Total Repayment Amount")
        ax.yaxis.set_major_formatter(y_formatter)
        plt.savefig('files\\{0}_{1}'.format(title, i))
        plt.show()



if __name__ == "__main__":
    try:
        print(small_df.head(1))
    except NameError:
        sdf = small_df = create_small_df(size=1000, cohort='dec_17')

    daily_sdf_pivot = convert_to_daily_pivot(small_df)        
    daily_cum_sdf = daily_sdf_pivot.cumsum(axis=0) 

    sdf = small_df['AmountPaid'].unstack(0).fillna(0).sort_index()
    monthly_sdf = sdf.groupby(pd.Grouper(freq='M')).sum()
    monthly_cum_sdf = monthly_sdf.cumsum(axis=0)


    #timeseries_length = (daily_sdf_pivot.index.max() - daily_sdf_pivot.index.min()).days


    AVERAGE_PAYMENT_FREQUENCY = timeseries_length / daily_sdf.astype(bool).sum(axis=0) 
    MEAN_PAYMENT_PER_DAY = daily_sdf.sum()/ timeseries_length
    
    ###### Plotting
    ## Daily Payments
    plot_daily_payments(daily_cum_sdf, batch_size=10, cohort_name='Dec 17', 
                        y_formatter=ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000)+'k')
                        )

    ## Smoothed Payments
    title = "Smoothed Cumulative Payments for 100 Random X850 Contracts in Jan 2019 Cohort"
    ax = daily_cum_sdf.rolling(window=30).mean().plot(figsize=(20,8), legend=False, title=title)
    ax.set_xlabel("Transaction Date")
    ax.set_ylabel("Total Repayment Amount")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000)+'k'))
    plt.savefig('files\\{}'.format(title))
    
    
    
    ## Percentage Payments
    cumulative_percent_sdf = create_percent_sdf(daily_sdf_pivot, cumulative=True, cohort='dec_17')
    # title = "Cumulative % Payments for 100 Random X850 Contracts in Jan 2019 Cohort"
    # ax = cumulative_percent_sdf.plot(figsize=(20,8), legend=False, title=title)
    # ax.yaxis.set_major_formatter()
    # ax.set_xlabel("Transaction Date")
    # ax.set_ylabel("files\\Total Repayment Amount % of Contract Value")
    # plt.savefig('files\\{}'.format(title))
    plot_daily_payments(cumulative_percent_sdf, batch_size=10, cohort_name='Dec 17',
                        y_formatter=ticker.FuncFormatter(lambda x, pos: '{:,.0%}'.format(x))
                        )
    
    
    
    ## Monthly Percentage Payments
    
    monthly_cum_perc_df = monthly_cum_sdf.divide(contract_values, axis=1)
    
    monthly_cum_perc_df.plot(figsize=(20,8), legend=False, title="Monthly Cumulative % Payments for 100 Random Contracts in Jan 2019 Cohort")
    ax.set_xlabel("Transaction Date")
    ax.set_ylabel("Total Repayment Amount % of Contract Value")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0%}'.format(x)))
    plt.savefig('files\\Monthly Cumulative percentage Payments for 100 Random Contracts in Jan 2019 Cohort')
    
    
    ## HISTAGRAM OF daily % rePAYMENTS
    
    payment_series_array =  percent_df.stack().to_numpy()
    daily_nonzero_percent_payments = payment_series_array[payment_series_array.nonzero()]
    
    f, ax = plt.subplots(figsize=(7, 5))
    plt.yscale('log')
    ax = sns.histplot(daily_nonzero_percent_payments, bins=100, )
    ax.set_title('Daily Payment Histogram')
    ax.set_xlabel('Payment %')
    ax.set_ylabel('Count')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0%}'.format(x)))
    plt.savefig('files\\Daily Payment Histogram.png')
    
    
    
    ## monthly HISTAGRAM OF daily % rePAYMENTS
    
    
    df = pd.merge(
        monthly_sdf.T,
        cdf,
        how='inner',
        left_index=True,
        right_index=True)
    
    percent_month_df = df.divide(df['TotalContractValue'], axis=0).drop(columns=['TotalContractValue'])
    month_payment_series_array =  percent_month_df.stack().to_numpy()
    monthly_nonzero_percent_payments = month_payment_series_array[month_payment_series_array.nonzero()]
    
    f, ax = plt.subplots(figsize=(7, 5))
    #plt.yscale('log')
    ax = sns.histplot(monthly_nonzero_percent_payments, bins=100, )
    ax.set_title('Monthly Payment Histogram')
    ax.set_xlabel('Payment %')
    ax.set_ylabel('Count')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0%}'.format(x)))
    plt.savefig('files\\Monthly Payment Histogram.png')
    
    
    ### Monthly bar plot of 100 contracts monthly payments
    monthly_sdf.plot(kind='bar')
    
    monthly_sdf[monthly_sdf.columns[0]].plot()
