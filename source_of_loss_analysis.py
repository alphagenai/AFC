# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:30:40 2021

@author: mark
"""


##loss analysis

import pandas as pd
import matplotlib.ticker as mtick

from matplotlib import pyplot as plt

from individual_analysis1 import create_percent_sdf
from calculate_days_dropped import calculate_days_dropped


def stacked_boxplot(ax,labels):
    plt.style.use('fivethirtyeight')
    
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    # .patches is everything inside of the chart
    for rect, label in zip(ax.patches, labels):
        # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        
        # The height of the bar is the data value and can be used as the label
        label_text = f'{height/10**6:,.2f}m - {label}'  # f'{height:.2f}' to format decimal values
        
        # ax.text(x, y, text)
        label_x = x + width / 2
        label_y = y + height / 2
    
        # plot only when height is greater than specified value
        if height > 0:
            ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8)
        
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)    
    ax.set_ylabel("Count", fontsize=18)
    ax.set_xlabel("Class", fontsize=18)
    plt.savefig('Source of losses.png')
    plt.show()
    
    

df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)

monthly_sdf = df.groupby(['ContractId',pd.Grouper(key='TransactionTS', freq='M',)])['AmountPaid'].sum()
daily_sdf = df.groupby(['ContractId',pd.Grouper(key='TransactionTS', freq='D',)]).sum()


monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
daily_sdf_pivot = daily_sdf['AmountPaid'].unstack(0).fillna(0)


monthly_cumulative_percent_sdf_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=True, cohort='dec_17')

daily_sdf_fullts = calculate_days_dropped(daily_sdf)

        
defaults = (monthly_sdf_pivot==0).astype('boolean') # total non-NaN: 36,000 incl Dec; 28,593 incl. Jan 18
paid  = (monthly_sdf_pivot!=0).astype('boolean')
fully_paid = monthly_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag
        

SQL = """
    SELECT c.ContractId,
        Price + AdditionalFee as TotalContractValue,     
        --c.RegistrationDate 
    FROM `afcproj.files_dupe.Contracts_20201117` c
    join `afcproj.files_dupe.{}_cohort` j
        on c.ContractId = j.ContractId
    """.format("dec_17")
cdf = pd.read_gbq(SQL,index_col='ContractId').astype('float64')

contract_ts = pd.merge(
    monthly_sdf_pivot.T,
    cdf,
    how='inner',
    left_index=True,
    right_index=True)

contract_values = contract_ts['TotalContractValue']

deposit = contract_values - contract_ts[contract_ts.columns[0]]
monthly_contractual_amount = (contract_values - deposit)/18.  # in AFC's spreadsheet they use approx. 19 months as contract term
deficit = monthly_sdf_pivot - monthly_contractual_amount
default_losses = deficit == -monthly_contractual_amount

deficit[default_losses].mask(fully_paid)
deficit[~default_losses].mask(fully_paid)

monthly_sdf_pivot.mask(fully_paid)

final_deficit = monthly_sdf_pivot.sum() - contract_values  


### by month
monthly_payments = monthly_sdf_pivot.sum(axis=1).to_frame('total_paid')

monthly_payments['contractual'] = 0
monthly_payments['contractual'].loc['jan-2018':'jun-2019'] =  (contract_values.sum() - monthly_payments['total_paid'].values[0]) /18.

monthly_payments.plot(kind='bar')

monthly_payments['max_electricity'] = monthly_payments.index.daysinmonth*55*1000

monthly_payments['deficit'] = monthly_payments['max_electricity'] - monthly_payments['total_paid'] 



### end-of-life

end_df = monthly_sdf_pivot.sum(axis=0).to_frame('total_paid').join(contract_values)
end_df['Deposit'] = monthly_sdf_pivot.iloc[0, :]
end_df['Paid during initial term'] = monthly_sdf_pivot.iloc[1:19, :].sum(axis=0)
end_df['Paid after initial term'] = monthly_sdf_pivot.iloc[19:, :].sum(axis=0)
end_df['Remains unpaid'] = end_df['TotalContractValue'] - end_df['total_paid']
end_df['Overpayments'] = end_df['Remains unpaid'][end_df['Remains unpaid'] < 0]
end_df['Remains unpaid'] = end_df['Remains unpaid'][end_df['Remains unpaid'] >= 0]

#end_df.sum().plot(kind='bar')

data_to_plot = end_df.sum().drop(['TotalContractValue', 'total_paid'])
labels = data_to_plot.index
ax = pd.DataFrame(data_to_plot).T.plot.bar(stacked=True, legend=False)
stacked_boxplot(ax, labels)

unfinished_contracts = end_df['Remains unpaid'] > 0

daily_cumulative_percent_sdf_pivot = create_percent_sdf(daily_sdf_pivot, cumulative=True, cohort='dec_17')
fully_paid = daily_cumulative_percent_sdf_pivot.shift(1) >= 1.0 #final payment is not included in fully paid flag

no_elec_pivot = daily_sdf_fullts['days_out_of_elec'].unstack(0).loc['2017-12-01':'2020-11-17'].mask(fully_paid.loc['2017-12-01':'2020-11-17'])
noelec_end = no_elec_pivot.iloc[-1]

end_df['Days without elec at end of contract'] = noelec_end


#daily_sdf_fullts.loc['1350828', :]

end_df['Days without elec at end of contract category'] = pd.cut(end_df['Days without elec at end of contract'], [0, 10, 30, 90, 1080])

s = end_df['Remains unpaid'].groupby(end_df['Days without elec at end of contract category']).sum()

ax = pd.DataFrame(s).T.plot.bar(stacked=True, legend=False)
labels = s.index

stacked_boxplot(ax, labels)