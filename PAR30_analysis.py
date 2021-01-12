# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:54:59 2021

@author: mark
"""

########## TRANSITION PROBABILITIES FROM PAR30 STATE

par30_daily_pivot = daily_sdf_fullts['PAR30+'].unstack(0).sort_index()


## completed contracts are converted to NaN
## be careful that the start and end dates of both dataframes is the same
fully_paid = daily_cumulative_percent_sdf_pivot.shift(1) >= 0.80 #final payment is not included in fully paid flag
par30_daily_pivot = par30_daily_pivot.mask(fully_paid).astype('boolean')

daily_cumulative_percent_sdf_pivot.loc[:,par30_daily_pivot.any(axis=0)].plot()
daily_cumulative_percent_sdf_pivot.loc[:,par30_daily_pivot.any(axis=0)].to_csv('PAR30_examples.csv')
par30_daily_pivot.loc[:, par30_daily_pivot.any(axis=0)].to_csv('Par_30_examples2.csv')

## seems right
#daily_cumulative_percent_sdf_pivot.loc['June-2018':'Aug-2018','1349716'].plot()

unconditional_daily_PE = par30_daily_pivot.sum().sum() / par30_daily_pivot.count().sum() 

par30_yesterday_pivot = par30_daily_pivot.shift(1)

transition_to_par30 = ~par30_yesterday_pivot & par30_daily_pivot 
transition_from_par30 = par30_yesterday_pivot & ~par30_daily_pivot 

transition_df = pd.concat([transition_to_par30.sum(), transition_from_par30.sum()], axis=1) #.to_csv('transition_probs.csv')

counterparty_events = transition_df.astype('bool').sum()  #recovery happens at least once to 413 counterparties
number_of_events = transition_df.sum()

recovery = (transition_df[0] == transition_df[1]) & transition_df[0].astype('bool')  
recovery.sum() # recovery happens EVERY TIME to 319 counterparties

probability_of_recovery = transition_df.sum()[1] / transition_df.sum()[0]  # 1,359/1,466 


""" 
TO DO: investigate sequences of par30 and how long they tend to be. 
hypothesis: the longer the par30, the less likely the recovery
"""

longest_period_of_no_elec = daily_sdf_fullts['days_out_of_elec'].unstack(0).max()

longest_period_if_par30 = longest_period_of_no_elec[longest_period_of_no_elec > 30]

sns.histplot(longest_period_if_par30 , bins=50 )

single_default = (transition_df[0] == 1) & (transition_df[1] == 0)
single_default[single_default]


########## Investigate sequences 

daily_sdf_fullts['par30_seqno'] = daily_sdf_fullts[daily_sdf_fullts['PAR30+']].groupby(
    ['ContractId','PAR30+']
    ).ngroup()

par30_sequence_length = daily_sdf_fullts.groupby(['ContractId', 'par30_seqno'])['days_out_of_elec'].max()

daily_sdf_fullts['TransactionTS'] = daily_sdf_fullts.index.get_level_values(1)
par30_sequence_end_date = daily_sdf_fullts.groupby(['ContractId', 'par30_seqno'])['TransactionTS'].max()
## when does the sequence happen after 80% paid? 
par30_amount_paid = pd.merge(daily_cumulative_percent_sdf_pivot.stack().to_frame().reset_index(),
         daily_sdf_fullts.reset_index(),
         left_on=['ContractId', 'level_0'], 
         right_on=['ContractId', 'TransactionTS'],
         )  ## column of % cumulative payment amount is unfortunately called 0