# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 00:12:12 2020

@author: mark
"""

import logging
import sys

import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.stats import beta

from individual_analysis1 import create_percent_sdf

sns.set_style('white')

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')


""" tODO: LIFETIME PD - the probability of entering PAR30 then never paying anything again """



def main():

    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    
    df = df.groupby(['ContractId', 'TransactionTS']).sum()
    
    daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
    monthly_sdf = daily_sdf.groupby(['ContractId',pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
    monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
    return monthly_sdf_pivot


def daily_default_full_timeseries():
    default = daily_sdf_fullts['PAR30+']
    default_event = (default.diff() != 0) & (daily_sdf_fullts['PAR30+']==True)
    recovery_event = (default.diff() != 0) & (daily_sdf_fullts['PAR30+']==False) & (
        (default.index.get_level_values(1).month != 12) & (default.index.get_level_values(1).year != 2017)
        )
    daily_sdf_fullts[recovery_event].to_csv('recoveries.csv')
    daily_sdf_fullts[default_event].to_csv('defaults.csv')



#############  FREQUENTIST

def how_many_true(df):
    return df.sum().sum()

def how_many(df):
    return df.count().sum()

class PDCalculator(object):
    """ TO DO: ADD BAYES TO THIS CLASS """
    def __init__(self,monthly_sdf_pivot,):
        self._point_estimate = None
        self._temporal_estimate = None
        self._counterparty_estimate = None
        self.PD_dict = {}
        self._monthly_sdf_pivot = monthly_sdf_pivot

    def data_prep(self, ):
        monthly_sdf_pivot = self._monthly_sdf_pivot
        monthly_cumulative_percent_sdf_pivot = create_percent_sdf(monthly_sdf_pivot, cumulative=True, cohort='dec_17')
        self._monthly_cumulative_percent_sdf_pivot = monthly_cumulative_percent_sdf_pivot
        
        defaults = (monthly_sdf_pivot==0).astype('boolean') # total non-NaN: 36,000 incl Dec; 28,593 incl. Jan 18
        paid  = (monthly_sdf_pivot!=0).astype('boolean')
        fully_paid = monthly_cumulative_percent_sdf_pivot.shift(1) >= 0.99 #final payment is not included in fully paid flag
        
        ## completed contracts are converted to NaN
        defaults = defaults.mask(fully_paid).astype('boolean')
        paid  = paid.mask(fully_paid).astype('boolean')
        
        #individual_uncond_PD = defaults.sum(axis=0, skipna=True)/defaults.count(axis=0)  #df.count() does not count NaNs
        #unconditional_PD = defaults.sum(axis=0, skipna=True).sum()/defaults.count(axis=0).sum()
        
        last_month_defaults = defaults.shift(1)
    
        ## ignore first 2 months as cannot default in origination month
        defaults = defaults.iloc[2:]  # 3,372 defaults in 28,214 total 
        last_month_defaults = last_month_defaults.iloc[2:]  #28,647
        paid = paid.iloc[2:] #24,752 of 28,124
        
        self._defaults = defaults
        self._last_month_defaults = last_month_defaults
        
        return defaults, last_month_defaults

    def calc_PD(self, kind="point_estimate"):
        
        monthly_sdf_pivot = self._monthly_sdf_pivot
        defaults, last_month_defaults = self.data_prep()
        
        """ TO DO: FIX THIS: """
        ### REMEMBER FALSE & NA = FALSE!!
        D_given_D = defaults & last_month_defaults  # 2,233 events
        D_given_ND = defaults & ~last_month_defaults # 1,139
        
        ND_given_D = ~defaults & last_month_defaults # 1,073
        ND_given_ND = ~defaults & ~last_month_defaults # 23,679
    
        NA_given_D = defaults.isna() & last_month_defaults # 0
        NA_given_ND = defaults.isna() & ~last_month_defaults # 523 ##FIX THIS
        
        logging.info('NA_given_D: {}'.format(how_many_true(NA_given_D)))
        logging.info('NA_given_ND: {}'.format(how_many_true(NA_given_ND)))
    
        ## for completeness
        D_given_NA = defaults & last_month_defaults.isna() # 0
        ND_given_NA = ~defaults & last_month_defaults.isna() # 0 
    
        logging.info('D_given_NA: {}'.format(how_many_true(D_given_NA)))
        logging.info('ND_given_NA: {}'.format(how_many_true(ND_given_NA)))
    
    
        if kind == "point_estimate":
            PD_dict = self.point_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND)
        self.PD_dict = PD_dict
        return PD_dict
        

    ## 1) MONTH OF DEFAULT DOES NOT MATTER
    

    def point_estimate(self, D_given_D, D_given_ND, ND_given_D, ND_given_ND): 
    
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
        self.PD_dict = PD_dict ## TO DO: Diferentiate between the different types of PD
        return PD_dict
        
    ## 2) MONTH MATTERS
    
    def temporal_estimate(self, D_given_D, D_given_ND, ND_given_D, ND_given_ND):
    
        total_given_D = D_given_D.sum(axis=1) + ND_given_D.sum(axis=1)
        total_given_ND = D_given_ND.sum(axis=1) + ND_given_ND.sum(axis=1)
    
        
    
        PD_given_D = D_given_D.sum(axis=1) / total_given_D 
        PD_given_ND = D_given_ND.sum(axis=1) / total_given_ND # = (1 - PD_given_D)
        PND_given_D = ND_given_D.sum(axis=1) / total_given_D # = (1 - PND_given_ND)
        PND_given_ND = ND_given_ND.sum(axis=1) / total_given_ND 
        
        PD_given_D + PD_given_ND + PND_given_D + PND_given_ND
        PD_dict = {'PD_given_D':PD_given_D, 
                   'PD_given_ND':PD_given_ND, 
                   'PND_given_D':PND_given_D, 
                   'PND_given_ND':PND_given_ND,
                   }
        self.PD_dict = PD_dict ## TO DO: Diferentiate between the different types of PD
        return PD_dict
    
        ## 3) COUNTERPARTY MATTERS
        

    def counterparty_estimate(self, D_given_D, D_given_ND, ND_given_D, ND_given_ND):
    
        total_given_D = D_given_D.sum(axis=0) + ND_given_D.sum(axis=0)
        total_given_ND = D_given_ND.sum(axis=0) + ND_given_ND.sum(axis=0)
    
        
    
        PD_given_D = D_given_D.sum(axis=0) / total_given_D 
        PD_given_ND = D_given_ND.sum(axis=0) / total_given_ND # = (1 - PD_given_D)
        PND_given_D = ND_given_D.sum(axis=0) / total_given_D # = (1 - PND_given_ND)
        PND_given_ND = ND_given_ND.sum(axis=0) / total_given_ND 
        
        PD_given_D + PD_given_ND + PND_given_D + PND_given_ND
        PD_dict = {'PD_given_D':PD_given_D, 
                   'PD_given_ND':PD_given_ND, 
                   'PND_given_D':PND_given_D, 
                   'PND_given_ND':PND_given_ND,
                   }
        self.PD_dict = PD_dict ## TO DO: Diferentiate between the different types of PD
        return PD_dict



#### BAYESIAN

class BayesianPDUpdater(object):
    def __init__(self, initial_params):
        self._parameter_dict = initial_params
        
    def update_logic(self, two_element_series,):
        # default & prev_default
        if two_element_series[0] & two_element_series[1]:
            self._parameter_dict["PD_given_D_alpha"] += 1
        # default & not prev_default
        elif two_element_series[0] & ~two_element_series[1]:
            self._parameter_dict["PD_given_ND_alpha"] += 1
        # no default & prev_default
        elif ~two_element_series[0] & two_element_series[1]:
            self._parameter_dict["PD_given_D_beta"] += 1
        # no default & no prev_default
        elif ~two_element_series[0] & ~two_element_series[1]:
            self._parameter_dict["PD_given_ND_beta"] += 1
    
    def update_PD(self, hist_df):
        hist_df.apply(self.update_logic, axis=1)
        return self._parameter_dict
    
def bayes_PD_updates(forecast_startdate, one_contract_id, defaults, last_month_defaults, 
                     PD_given_D_alpha, PD_given_D_beta, PD_given_ND_alpha, PD_given_ND_beta):
    """ This is where a HMM could come in - classify customers into credit buckets for conditional PDs """

    hist_d = defaults.loc[:forecast_startdate, one_contract_id]
    hist_prev_d = last_month_defaults.loc[:forecast_startdate, one_contract_id]

    hist_df = pd.concat([hist_d, hist_prev_d], axis=1)    

    print(hist_df)
    
    labels = hist_df.apply(label_logic, axis=1, parameter_dict=_parameter_dict)
    counts = labels.groupby(labels).count()
    
    ### DOES THIS ACTUALLY WORK??
    # empcounts = pd.Series(data=[0,0,0,0], index=[['D_given_D',
    #                                            'ND_given_D',
    #                                            'D_given_ND',
    #                                            'ND_given_ND',]])
    # empcounts += counts

    # empcounts.add(counts, fill_value=0)
    
    PD_given_D_alpha += counts['D_given_D']
    PD_given_D_beta += counts['ND_given_D']
    
    PD_given_ND_alpha += counts['D_given_ND']
    PD_given_ND_beta += counts['ND_given_ND']
    
    parameter_dict = {'PD_given_D_alpha':PD_given_D_alpha,
                      'PD_given_D_beta':PD_given_D_beta,
                      'PD_given_ND_alpha':PD_given_ND_alpha,
                      'PD_given_ND_beta':PD_given_ND_beta,
                      }
    return parameter_dict

    def calc_mean(self):
        mean = self._parameter_dict['PD_given_D_alpha'] / (
            self._parameter_dict['PD_given_D_alpha'] + self._parameter_dict['PD_given_D_beta'])



def prior_analysis():
    c_PD_dict = counterparty_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND)
    t_PD_dict = temporal_estimate(D_given_D, D_given_ND, ND_given_D, ND_given_ND)

    cdf = pd.DataFrame(c_PD_dict)
    tdf = pd.DataFrame(t_PD_dict)
    cdf = cdf[['PD_given_D', 'PD_given_ND']]
    tdf = tdf[['PD_given_D', 'PD_given_ND']]
    

    cdf = cdf[~cdf.isna().any(axis=1)]  ## Look into WHY we get these
    #df = df[~(df==1.).any(axis=1)]
    #df = df[~(df==0.0).any(axis=1)]

    df = pd.concat([cdf, tdf,], axis=0)
        
    plot_prior_histogram(cdf, title='Counterparty PDs')
    plot_prior_histogram(tdf, title='Temporal PDs')
    plot_prior_histogram(df, title='All PDs')


def plot_prior_histogram(df, title):
    df.plot(kind='hist', bins=30, title=title)
    plt.savefig('files\{}.png'.format(title))

def plot_beta(a, b, ax=None):
    x = np.arange (0, 1, 0.01)
    y = beta.pdf(x,a,b, )
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x,y)
        


if __name__ == "__main__":

    monthly_sdf_pivot = main()
    pd_calc = PDCalculator(monthly_sdf_pivot)
    PD_dict = pd_calc.calc_PD()
    
    one_contract_id = monthly_sdf_pivot.columns[1]  # nice and switchy
    #one_contract_id = monthly_sdf_pivot.columns[3]  # same as lattice
    #one_contract_id = monthly_sdf_pivot.columns[0]  # lots of defaults

    forecast_startdate = '2019-12-31'


    hist_d = pd_calc._defaults.loc[:forecast_startdate, one_contract_id]
    hist_prev_d = pd_calc._last_month_defaults.loc[:forecast_startdate, one_contract_id]

    hist_df = pd.concat([hist_d, hist_prev_d], axis=1)    
    
    initial_params = {'PD_given_D_alpha':1.1,
                      'PD_given_D_beta':6,
                      'PD_given_ND_alpha':1.1,
                      'PD_given_ND_beta':3,
                      }
    
    bpdu = BayesianPDUpdater(initial_params)
    bpdu.update_PD(hist_df)
    
    print(bpdu._parameter_dict)

    # parameter_dict = bayes_PD_updates(forecast_startdate, one_contract_id, 
    #                                   pd_calc._defaults, pd_calc._last_month_defaults, 
    #                                   PD_given_D_alpha, PD_given_D_beta, PD_given_ND_alpha, 
    #                                   PD_given_ND_beta)
    


    plot_beta(bpdu._parameter_dict['PD_given_D_alpha'], 
                   bpdu._parameter_dict['PD_given_D_beta'])

    plot_beta(bpdu._parameter_dict['PD_given_ND_alpha'], 
                   bpdu._parameter_dict['PD_given_ND_beta'])



