# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:52:09 2020

@author: mark
"""


# Import pyplot, figures inline, set style, plot pairplot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


categorical_columns = [ 'Region',
                         'Town',
                         'Occupation',]
binary_columns = ['MainApplicantGender',]

numerical_columns = ['Age', 'TotalContractValue',]


df = pd.merge(all_features,
         last_payment,
         left_index=True,
         right_index=True,
         )


cat_cols = categorical_columns + binary_columns


cat_df_to_plot = df[cols+[pd.Timestamp('2020-11-30 00:00:00')]]
payment_df_to_plot = all_features.drop(columns=cols)
sns.pairplot(payment_df_to_plot,) #hue='Age');

sns.catplot(x="Occupation", y=pd.Timestamp('2020-11-30 00:00:00'), kind="swarm", data=cat_df_to_plot)