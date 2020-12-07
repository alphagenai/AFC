# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 00:33:48 2020

@author: mark
"""


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




def feature_regression(input_df):
    ### assemble 6 months of payments features
    
    #first_6_month_payments = small_df.sort_index().xs(slice('2019-01-01', '2019-06-01'), level='TransactionTS', drop_level=False)
    #monthly_6m = first_6_month_payments.groupby(pd.Grouper(freq='M')).sum()
    
    
    first_6_month_payments = monthly_cumulative_percent_sdf['2019-01-01':'2019-06-01']
    last_payment = monthly_cumulative_percent_sdf.loc[monthly_cumulative_percent_sdf.index.max()]
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
    
    categorical_columns = [ 'MainApplicantGender',
                            'Region',
                            'Town',
                            'Occupation',]
    numerical_columns = ['Age', 'TotalContractValue',]
    
    preprocessor = make_column_transformer(
        (OneHotEncoder(drop='if_binary'), categorical_columns),
    #    (LabelEncoder(), categorical_columns),  #gives error
        remainder='passthrough'
        )
    
    model = RandomForestRegressor()
    
    pipe = make_pipeline(
        preprocessor,
        SimpleImputer(strategy='mean'),
        model,
        )
    
    
    X = all_features.sort_index()
    y = last_payment.sort_index()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    start = time.process_time()

    pipe.fit(X,y)
    print('time taken to fit model: {:.2f}'.format(time.process_time() - start))
    
    fig, ax = plt.subplots()
    ax = sns.histplot(model.predict(X)-y)
    plt.show()
    
    
    #model.score(X,y)



if __name__ == "__main__":
    
    small_df = create_small_df(size=1000)
    cum_perc_df = create_cumulative_percent_sdf(small_df)

'''

num_proc = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

cat_proc = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='missing'),
    OneHotEncoder(handle_unknown='ignore'))

preprocessor = make_column_transformer((num_proc, ('feat1', 'feat3')),
                                       (cat_proc, ('feat0', 'feat2')))
"""or"""
column_trans = ColumnTransformer(
    [
        ("binned_numeric", KBinsDiscretizer(n_bins=10),
            ["VehAge", "DrivAge"]),
        ("onehot_categorical", OneHotEncoder(),
            ["VehBrand", "VehPower", "VehGas", "Region", "Area"]),
        ("passthrough_numeric", "passthrough",
            ["BonusMalus"]),
        ("log_scaled_numeric", log_scale_transformer,
            ["Density"]),
    ],
    remainder="drop",
)
X = column_trans.fit_transform(df)
"""or"""
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, categorical_columns),
     ('num', numerical_pipe, numerical_columns)])


mapper = DataFrameMapper([
        (label_columns, LabelEncoder()),
        (float_columns,  [Imputer(missing_values='NaN', strategy='mean', axis=0), StandardScaler()],),
        (binary_columns, None),
        (integer_columns, OneHotEncoder())
        ])
 
_classifier = xgb.sklearn.XGBClassifier()#objective='multi:softmax', subsample=0.7)
pipeline = Pipeline([
                    ('featurize', mapper),
                    ('cls', _classifier ),
                    ])
