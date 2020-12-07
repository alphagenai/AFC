# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 00:33:48 2020

@author: mark
"""


import pandas as pd

from individual_analysis1 import create_small_df, create_cumulative_percent_sdf

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.ensemble import RandomForestRegressor
#from pandas.tools.plotting import scatter_matrix
#from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import time

from matplotlib import pyplot as plt
import seaborn as sns


from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, TheilSenRegressor, TweedieRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



"""

TO DO: Change from dates to monthdiffs so that all cohorts can be run together

"""

def feature_regression(input_df, models_list):   ## input df is currently a montly cumulative percentage
    ### assemble 6 months of payments features    
    
    first_6_month_payments = input_df['2019-01-01':'2019-06-01']
    last_payment = input_df.loc[input_df.index.max()]
    last_payment.sort_values().plot(kind='bar')
    payments_feature_df = first_6_month_payments.T
    
    ## assemble other features
    
    contract_sql = """
        SELECT c.ContractId,
                c.MainApplicantGender, 
                c.Age, 
                c.Region,
                c.Town,
                c.Occupation, 
            Price + AdditionalFee as TotalContractValue,     
        FROM `afcproj.files_dupe.Contracts_20201117` c
        join `afcproj.files_dupe.jan_19_cohort` j
            on c.ContractId = j.ContractId
            """
    cfdf = pd.read_gbq(contract_sql,index_col='ContractId') #.astype('float64')
    
    all_features = pd.merge(payments_feature_df,
             cfdf,
             how='inner',
             left_index=True,
             right_index=True).sort_index()
    
    categorical_columns = [ 'Region',
                            'Town',
                            'Occupation',]
    numerical_columns = ['Age', 'TotalContractValue',]
    binary_columns = ['MainApplicantGender',]
    
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), categorical_columns), #one hot is all zero when unknown labels
        (OneHotEncoder(drop='if_binary'), binary_columns), 
    #    (LabelEncoder(), categorical_columns),  #gives error
        remainder='passthrough'
        )
    
    
    
    X = all_features.sort_index()
    y = last_payment.sort_index()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)


    for model in models_list:

        pipe = make_pipeline(
        preprocessor,
        SimpleImputer(strategy='mean'),
        model,
        )
    


        start = time.process_time()
    
        pipe.fit(X_train,y_train)
        print('time taken to fit model: {:.2f}'.format(time.process_time() - start))
        
        pred_error = pipe.predict(X_test)-y_test
        
        fig, ax = plt.subplots()
        ax = sns.histplot(pred_error)
        plt.title(model)
        plt.show()
        
        
        #model.score(X,y)



if __name__ == "__main__":
    
    try:
        print(small_df.head(1))
    except NameError:
        small_df = create_small_df(size=1000)
    sdf = small_df['AmountPaid'].groupby([small_df.index.get_level_values(1).date, small_df.index.get_level_values(0)]).sum().unstack('ContractId').fillna(0).sort_index()
    sdf.index = pd.to_datetime(sdf.index)
    monthly_sdf = sdf.groupby(pd.Grouper(freq='M')).sum()
    cum_perc_df = create_cumulative_percent_sdf(monthly_sdf)
    
    models = [SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
              SVR(kernel='linear', C=100, gamma='auto'),
              #SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1),  ## seems to take a very long time
              RandomForestRegressor(),
              LinearRegression(),
              ]
    feature_regression(cum_perc_df, models)

