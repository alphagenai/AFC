# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:44:58 2020

@author: mark
"""


"""

************** THIS CODE SHOULD NOT BE CHANGED ************

THIS CODE SHOULD SERVE AS A BENCHMARK/CHECK AGAINST FUTURE CHANGES IN OTHER CODE ONLY

***********************************************************

"""


import google_sa_auth
import pandas as pd
import random


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import time

from matplotlib import pyplot as plt
import seaborn as sns


from sklearn.model_selection import cross_val_predict

sns.set()


def month_diff(a, b):
    return 12 * (a.dt.year - b.dt.year) + (a.dt.month - b.dt.month)


def create_small_df(size=100, limit=False, use_monthdiff=False, random_seed=42):    
    SQL = """ 
        Select p.TransactionTS,
            p.AmountPaid,
            p.ContractId, 
            c.RegistrationDate
        FROM afcproj.files_dupe.Payments_2020_11_17 p
        inner join afcproj.files_dupe.jan_19_cohort j
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
            c.RegistrationDate
        FROM afcproj.files_dupe.Adjustments_2020_11_17 a
        inner join afcproj.files_dupe.jan_19_cohort j
            on a.ContractId = j.ContractId
        inner join afcproj.files_dupe.Contracts_20201117 c
            on c.ContractId = j.ContractId
    
        WHERE a.BalanceChangeType = 'MANUAL'
            and (c.Product = 'X850'
            or c.Product = 'X850 Plus')
    
            """
    
    #for contractID in cohort:
    if limit:
        SQL = SQL + " LIMIT {}".format(limit)
    df = pd.read_gbq(SQL,) #chunksize=10000) #chunksize doesnt work
    
    if use_monthdiff:
        df['monthdiff'] = month_diff(df['TransactionTS'].dt.tz_localize(None), df['RegistrationDate']).clip(0,None)
        df = df.groupby(['ContractId','monthdiff']).sum()
    
    else:
        
        df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S')
        
        df = df.set_index(['ContractId','TransactionTS'])
                  
    df = df.astype('float64', errors='ignore')  ## datetime columns cause errors
    df = reduce_df_size(df, size=size, random_seed=random_seed)
    return df

def reduce_df_size(df, size, random_seed=42):
    random.seed(a=random_seed)        
    sample_random_IDs = random.sample(df.index.get_level_values(0).unique().values.tolist(), k=size,)
    
    small_df = df.loc[sample_random_IDs]   # see which IDs --> small_df.index.get_level_values(0).unique()
    return small_df

def create_percent_sdf(input_df, cumulative=True, use_monthdiff=False):
    
    
    ### Get Contract info
    
    SQL = """
        SELECT c.ContractId,
            Price + AdditionalFee as TotalContractValue,     
            --c.RegistrationDate 
        FROM `afcproj.files_dupe.Contracts_20201117` c
        join `afcproj.files_dupe.jan_19_cohort` j
            on c.ContractId = j.ContractId
        """
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
    
    return return_df

def convert_to_daily(small_df ):
    sdf = small_df['AmountPaid'].unstack(0).fillna(0).sort_index()
    
    daily_sdf = sdf.groupby(sdf.index.date).sum() #all payments in one day are grouped together
    return daily_sdf






def create_features(input_df, target_monthdiff, calc_cumsum):
    ### assemble 6 months of payments features    
    
    first_6_month_payments = input_df.iloc[0:6]  #slicing on index
    
    if calc_cumsum:
        input_df = input_df.cumsum(axis=0)
    
    target_payment = input_df.loc[target_monthdiff].sort_values()   # some cumulative payments are greater than 1 ?

    ##Plot what these target payments look like
    ## A priori spread of final payments is pretty linear
    #target_payment.plot(kind='bar')
    
    payments_feature_df = first_6_month_payments.T
    
    ## assemble other features
    
    contract_sql = """
        SELECT c.ContractId,
                c.MainApplicantGender, 
                c.Age, 
                c.Region,
                c.Town,
                c.Occupation, 
                c.Product,
            Price + AdditionalFee as TotalContractValue,     
        FROM `afcproj.files_dupe.Contracts_20201117` c
        join `afcproj.files_dupe.jan_19_cohort` j
            on c.ContractId = j.ContractId
            """
    cfdf = pd.read_gbq(contract_sql, index_col='ContractId', dialect='standard')  #.astype('float64')
    
    all_features = pd.merge(payments_feature_df,
             cfdf,
             how='inner',
             left_index=True,
             right_index=True).sort_index()
    return all_features, target_payment  


def feature_regression(input_df, models_list, target_monthdiff=18, calc_cumsum=False):   ## input df is currently a montly cumulative percentage
    

    all_features, target_payment = create_features(input_df, target_monthdiff, calc_cumsum)
    categorical_columns = [ 'Region',
                            'Town',
                            'Occupation',
                            'Product',
                            ]
    numerical_columns = ['Age', 'TotalContractValue',]
    binary_columns = ['MainApplicantGender',]
    
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), categorical_columns), #one hot is all zero when unknown labels
        (OneHotEncoder(drop='if_binary'), binary_columns), 
    #    (LabelEncoder(), categorical_columns),  #gives error
        remainder='passthrough'
        )
    
    
    
    X = all_features.sort_index()
    y = target_payment.sort_index()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)


    for model in models_list:

        pipe = make_pipeline(
        preprocessor,
        SimpleImputer(strategy='mean'),
        model,
        )
    


        start = time.process_time()
        try:
            pipe.fit(X_train,y_train)
        except Exception as e:
            print(e)
            return pipe, all_features   # for debugging
        print('time taken to fit model: {:.2f}'.format(time.process_time() - start))

        plot_error_histogram(pipe, model, X_test, y_test)
        plot_actual_v_predicted(pipe, model, X_test, y_test)


    return (pipe, model, X, y), None
        

def plot_error_histogram(pipe, model, X_test, y_test):
    pred_error = pipe.predict(X_test)-y_test
    
    fig, ax = plt.subplots()
    ax = sns.histplot(pred_error)
    plt.title('Out of Sample Errors for {}'.format(model))
    plt.show()
    plt.savefig('{} error histogram'.format(model))
    

def plot_actual_v_predicted(pipe, model, X, y):

    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:
    predicted = cross_val_predict(pipe, X, y, cv=10)
    
    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.title('Actual vs Predicted for model {}'.format(model))
    plt.show()
    plt.savefig('actual vs predicted for {}'.format(model))


if __name__ == "__main__":
    try:  #why isnt this working?
        print(small_df.head(1))
    except NameError:
        print("SMALL_DF NOT FOUND")
        small_df = create_small_df(size=1000, use_monthdiff=True, random_seed=42)
        
    monthly_sdf = small_df['AmountPaid'].unstack('ContractId').fillna(0).sort_index()

    ##using monthdiff appears to make the model worse - WHY???
    df_for_model = create_percent_sdf(monthly_sdf, use_monthdiff=True, cumulative=False)
    
    
    
    models = [LinearRegression(),
              ]
    
    (pipe, model,X, y), feature_df_on_error = feature_regression(df_for_model, models, calc_cumsum=True)
    
    
    categorical_columns = [ 'Region',
                             'Town',
                             'Occupation',
                             'Product',
                             ]
    binary_columns = ['MainApplicantGender',]
    
    numerical_columns = ['Age', 'TotalContractValue',]
    
    all_features, target_payment = create_features(df_for_model, target_monthdiff=18, calc_cumsum=True)
    
    df = pd.merge(all_features,
             target_payment,
             left_index=True,
             right_index=True,
             )
    
    
    cat_cols = categorical_columns + binary_columns
    
    
    cat_df_to_plot = df[cat_cols+[18]]
    payment_df_to_plot = all_features.drop(columns=cat_cols)
    sns.pairplot(df[list(range(0,6))+[18,]], height=6) #hue='Age');
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('files\\Pairplot.png')
    
    
    ## Super Interesting! - Region/Covid
    g = sns.catplot(x="Region", y=18, kind="swarm", data=cat_df_to_plot, height=8)
    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=30)
    
    plt.gcf().subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig('files\\Region feature.png')
    
        
    ## Also Interesting
    sns.catplot(x="TotalContractValue", y=18, kind="swarm", 
                data=df, height=10)
    plt.xlabel('Contract Value', fontsize=12)
    plt.ylabel('Amount Repaid at Month 18', fontsize=12)
    plt.gcf().subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig('files\\Total Contract Value Feature.png')
