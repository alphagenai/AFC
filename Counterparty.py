# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:05:43 2021

@author: mark
"""

import pandas as pd

from scipy.stats import beta

from calc_PD import BayesianPDUpdater, PDCalculator

class Counterparty(object):
    def __init__(self, ContractId, total_contract_value=None):
        self.ContractId = ContractId
        self.total_contract_value = total_contract_value
        self.alpha = 1
        self.beta = 1
        self.PD = 0.055
        self.RSQ = 0.3
        
        self.PD_given_D_alpha = 1.1
        self.PD_given_D_beta = 6
        self.PD_given_ND_alpha = 1.1
        self.PD_given_ND_beta = 3
        
        self.params = {'PD_given_D_alpha':1.1,
                      'PD_given_D_beta':6,
                      'PD_given_ND_alpha':1.1,
                      'PD_given_ND_beta':3,
                      }
        
        
    def percent_timeseries(self, startdate=None, enddate=None):
        raise NotImplementedError()
        
    def update_PD_params(self, monthly_df_pivot, forecast_startdate, pd_calc=None):
        """ updates alpha and beta using calc_PD """
    
        bpdu = BayesianPDUpdater(self.params)
        
        if pd_calc is None:
            pd_calc = PDCalculator(monthly_df_pivot)    
            defaults, last_month_defaults = pd_calc.data_prep()
        else:
            defaults, last_month_defaults = pd_calc._defaults, pd_calc._last_month_defaults

        hist_d = defaults.loc[:forecast_startdate, self.ContractId]
        hist_prev_d = last_month_defaults.loc[:forecast_startdate, self.ContractId]
        hist_df = pd.concat([hist_d, hist_prev_d], axis=1)    
            
        self.params = bpdu.update_PD(hist_df)
        
    def update_bayesian_mean_PDs(self, monthly_df_pivot, forecast_startdate, pd_calc=None):
        self.update_PD_params(monthly_df_pivot, forecast_startdate, pd_calc=pd_calc)
    
        self.PD_given_D = beta.mean(self.params['PD_given_D_alpha'], self.params['PD_given_D_beta'])
        self.PND_given_D = 1 - self.PD_given_D 
        self.PD_given_ND = beta.mean(self.params['PD_given_ND_alpha'], self.params['PD_given_ND_beta'])
        self.PND_given_ND = 1 - self.PD_given_ND 
        
        self.PD_dict = {'PD_given_D':self.PD_given_D, 
                   'PD_given_ND':self.PD_given_ND, 
                   'PND_given_D':self.PND_given_D, 
                   'PND_given_ND':self.PND_given_ND,
                   }





if __name__ == "__main__":

    from calc_PD import main
    from scipy.stats import beta
    
    counterparty = Counterparty('1349817')
    
    monthly_sdf_pivot = main()
    counterparty.update_bayesian_mean_PDs(monthly_sdf_pivot, '1/1/2019')
    
    print(counterparty.PD_dict)