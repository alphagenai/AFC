# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:54:59 2021

@author: mark
"""


from calc_PD import how_many_true

for cutoff in [0.8, 0.99]:
    par30_daily_pivot = daily_sdf_fullts['PAR30+'].unstack(0).sort_index()

    ## completed contracts are converted to NaN
    ## be careful that the start and end dates of both dataframes is the same
    fully_paid = daily_cumulative_percent_sdf_pivot.shift(1) >= cutoff #final payment is not included in fully paid flag
    par30_daily_pivot = par30_daily_pivot.mask(fully_paid).astype('boolean')
    
    daily_cumulative_percent_sdf_pivot.loc[:,par30_daily_pivot.any(axis=0)].plot()
    daily_cumulative_percent_sdf_pivot.loc[:,par30_daily_pivot.any(axis=0)].to_csv('PAR30_examples.csv')
    par30_daily_pivot.loc[:, par30_daily_pivot.any(axis=0)].to_csv('Par_30_examples2.csv')
    
    ## seems right - this guy has a break in paying for July/early august
    #daily_cumulative_percent_sdf_pivot.loc['June-2018':'Aug-2018','1349716'].plot()
    
    unconditional_daily_PE = par30_daily_pivot.sum().sum() / par30_daily_pivot.count().sum() 
    
    
    transition_df = pd.concat([transition_to_par30.sum(), transition_from_par30.sum()], axis=1) #.to_csv('transition_probs.csv')
    
    counterparty_events = transition_df.astype('bool').sum()  #recovery happens at least once to 413 counterparties
    number_of_events = transition_df.sum()
    print('cutoff {}: number of events: {}'.format(cutoff, number_of_events))
    
    recovery = (transition_df[0] == transition_df[1]) & transition_df[0].astype('bool')  
    recovery.sum() # recovery happens EVERY TIME to 319 counterparties
    
    probability_of_recovery = transition_df.sum()[1] / transition_df.sum()[0]  # 1,359/1,466 
    print('cutoff {}, probability_of_recovery: {}'.format(cutoff, probability_of_recovery))

    unconditional_probability_of_PAR30_given_paying = transition_df.sum()[0]/total_days_of_nonPar30


def transition_probs(par30_daily_pivot):
    par30_yesterday_pivot = par30_daily_pivot.shift(1)
    
    transition_to_par30 = ~par30_yesterday_pivot & par30_daily_pivot 
    transition_from_par30 = par30_yesterday_pivot & ~par30_daily_pivot 
    stay_in_non_par30 = ~par30_yesterday_pivot & ~par30_daily_pivot  # 816,190
    stay_in_par30 = par30_yesterday_pivot & par30_daily_pivot  # 82,740
    
    total_days_of_nonPar30 = (~par30_daily_pivot).sum().sum() #818,609

    """ TO DO: FIX THIS: """
    ### REMEMBER FALSE & NA = FALSE!!
    P3_given_P3 = par30_daily_pivot & par30_yesterday_pivot  
    P3_given_NP3 = par30_daily_pivot & ~par30_yesterday_pivot  
    
    logging.info('P3_given_P3: {}'.format(how_many_true(P3_given_P3)))
    logging.info('P3_given_NP3: {}'.format(how_many_true(P3_given_NP3)))

    NP3_given_P3 = ~par30_daily_pivot & par30_yesterday_pivot  
    NP3_given_NP3 = ~par30_daily_pivot & ~par30_yesterday_pivot  

    logging.info('NP3_given_P3: {}'.format(how_many_true(NP3_given_P3)))
    logging.info('NP3_given_NP3: {}'.format(how_many_true(NP3_given_NP3)))

    NA_given_P3 = par30_daily_pivot.isna() & par30_yesterday_pivot # 
    NA_given_NP3 = par30_daily_pivot.isna() & ~par30_yesterday_pivot # 

    logging.info('NA_given_P3: {}'.format(how_many_true(NA_given_P3)))
    logging.info('NA_given_NP3: {}'.format(how_many_true(NA_given_NP3)))
    return point_estimate(P3_given_P3, P3_given_NP3, NP3_given_P3, NP3_given_NP3)

def point_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND): 
    """ from calc_PD """
    total_given_D = how_many_true(D_given_D) + how_many_true(ND_given_D)
    total_given_ND = how_many_true(D_given_ND) + how_many_true(ND_given_ND)

    PD_given_D = how_many_true(D_given_D) / total_given_D 
    PD_given_ND = how_many_true(D_given_ND) / total_given_ND # = (1 - PD_given_D)
    PND_given_D = how_many_true(ND_given_D) / total_given_D # = (1 - PND_given_ND)
    PND_given_ND = how_many_true(ND_given_ND) / total_given_ND 

    logging.info('PD_given_D : {} / {}'.format(how_many_true(D_given_D) , total_given_D))    
    logging.info('PD_given_ND : {} / {}'.format(how_many_true(D_given_ND) , total_given_ND))    
    logging.info('PND_given_D : {} / {}'.format(how_many_true(ND_given_D) , total_given_D))    
    logging.info('PND_given_ND : {} / {}'.format(how_many_true(ND_given_ND) , total_given_ND))    

   
    PD_dict = {'PD_given_D':PD_given_D, 
               'PD_given_ND':PD_given_ND, 
               'PND_given_D':PND_given_D, 
               'PND_given_ND':PND_given_ND,
               }
    PD_dict = PD_dict ## TO DO: Diferentiate between the different types of PD
    return PD_dict

def calc_monthly_transition_probs(PD_dict):
    monthly_transition_dict = {}
    for k, v in PD_dict.items():
        print(k)
        if k in ['PD_given_ND', 'PND_given_D']:
            monthly_prob = transition_prob_to_monthly(v)
            print(monthly_prob*100)
            monthly_transition_dict[k] = monthly_prob()
    monthly_transition_dict['PD_given_D'] = 1 - 'PD_given_ND'
    monthly_transition_dict['PND_given_ND'] = 1 - 'PND_given_D'
    monthly_transition_dict.to_pickle('monthly_transition_probs.pkl')
    return monthly_transition_dict


def transition_prob_to_monthly(prob):
    """ At least one transition in a month == 1 - probability(no transition)"""
    return 1 - ((1-prob)**30)

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