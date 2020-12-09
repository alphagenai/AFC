# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:52:09 2020

@author: mark
"""


# Import pyplot, figures inline, set style, plot pairplot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from AFC_prototype_model3 import create_features


all_features, target_payment = create_features(df_for_model, target_monthdiff=18, calc_cumsum=True)


categorical_columns = [ 'Region',
                         'Town',
                         'Occupation',
                         'Product',
                         ]
binary_columns = ['MainApplicantGender',]

numerical_columns = ['Age', 'TotalContractValue',]


df = pd.merge(all_features,
         target_payment,
         left_index=True,
         right_index=True,
         )


cat_cols = categorical_columns + binary_columns


cat_df_to_plot = df[cat_cols+[18]]
payment_df_to_plot = all_features.drop(columns=cat_cols)
sns.pairplot(df[list(range(0,6))+[18,]], height=6) #hue='Age');
plt.savefig('Pairplot.png')

#sns.catplot(x="Occupation", y=pd.Timestamp('2020-11-30 00:00:00'), kind="swarm", data=cat_df_to_plot)

#sns.stripplot(x="Occupation", y=pd.Timestamp('2020-11-30 00:00:00'), data=cat_df_to_plot)

## Super Interesting!
fig, ax = plt.subplots()
#sns.stripplot(x="Region", y=18, data=cat_df_to_plot)
g = sns.catplot(x="Region", y=18, kind="swarm", data=cat_df_to_plot, height=8)
g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=30)
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('Region feature.png')



#sns.stripplot(x="TotalContractValue", y=pd.Timestamp('2020-11-30 00:00:00'), data=cat_df_to_plot)
## Also Interesting
sns.catplot(x="TotalContractValue", y=18, kind="swarm", 
            data=df, height=25)
plt.savefig('Total Contract Value Feature.png')

## Not interesting - all X850
# sns.catplot(x="Product", y=18, kind="swarm", 
#             data=df, height=25)
# plt.savefig('Product.png')
