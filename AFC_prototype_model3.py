# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 00:33:48 2020

@author: mark
"""


import pandas as pd

from individual_analysis1 import create_small_df, create_percent_sdf

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


from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, TheilSenRegressor, TweedieRegressor, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



"""

TO DO: 
    1. Change from dates to monthdiffs so that all cohorts can be run together
    2. WHy is the monthdiff 0 column = NaN ?

"""

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


        pred_error = pipe.predict(X_test)-y_test
        
        fig, ax = plt.subplots()
        ax = sns.histplot(pred_error)
        plt.title('Out of Sample Errors for {}'.format(model))
        plt.show()
        plt.savefig('{} error histogram'.format(model))
        
        
    return (pipe, X, y), None
        
"""
##cant plot "the regression" through time as we dont regress thru time
def plot_in_sample(pipe, X, y):
    ## to do: show time series and regression line
    X = X.iloc[0:20]
    y = y.iloc[0:20]
    cum = X[X.columns[0:6]].cumsum(axis=1)
    cum.plot()
    y_pred = pipe.predict(X)
    fig,ax = plt.subplots()
    for col in range(0,6):
        plt.scatter(X[col], y, color='black')
        plt.scatter(X[col], y_pred, color='red')
"""

if __name__ == "__main__":
    try:  #why isnt this working?
        print(small_df.head(1))
    except NameError:
        print("SMALL_DF NOT FOUND")
        small_df = create_small_df(size=1000, use_monthdiff=True)
    monthly_sdf = small_df['AmountPaid'].unstack('ContractId').fillna(0).sort_index()

    ##using monthdiff appears to make the model worse - WHY???
    df_for_model = create_percent_sdf(monthly_sdf, use_monthdiff=True, cumulative=False)
    
    
    
    models = [RandomForestRegressor(),
              LinearRegression(),
              #SGDRegressor(), # not working
              #SGDRegressor(loss='log')  # not working
              ]
    
    ## Need to rework the target column if not using cumulative percentages
    (pipe, X, y), feature_df_on_error = feature_regression(df_for_model, models, calc_cumsum=True)


    #cum_df_for_model = create_percent_sdf(monthly_sdf, use_monthdiff=True, cumulative=True)
    #(pipe, X, y), feature_df_on_error = feature_regression(cum_df_for_model, models, calc_cumsum=False)
