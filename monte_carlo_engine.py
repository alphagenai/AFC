# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:30:44 2021

@author: mark
"""



import numpy as np
import pandas as pd

from numpy import random
from numpy import sqrt, exp

from scipy.stats import norm as scinorm
from scipy.stats import beta

from Counterparty import Counterparty



def main():
    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    
    df = df.groupby(['ContractId', 'TransactionTS']).sum()
    
    daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
    monthly_sdf = daily_sdf.groupby(['ContractId',pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
    monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)

    RSQ = 0.3
    PD = 0.055
    recovery_prob = 0.4
    
    d = (1-exp(-50*PD)) / (1-exp(-50))
    AIRB_RSQ = 0.12*(d) + 0.24*(1 - d)
    
    forecast_months = ['June', 'July', 'Aug']
    forecast_date = '31/12/2019'
    
    portfolio = {}
    for cid in df.index.get_level_values(0).unique():
        portfolio[j] = Counterparty(cid)
        portfolio[j].update_probabilities(forecast_date)
    
    
        #for i, month in enumerate(forecast_months):
            # random.seed(i)
            # state_of_the_world = random.standard_normal(1)
            # random.seed(i)
            # individual_risk = random.standard_normal(1000)



    def monthly_forecast(counterparty, state_of_the_world, individual_risk):
        i = 1
        
        individual_latent_factors = sqrt(counterparty.RSQ) * state_of_the_world + sqrt(1-RSQ)*individual_risk
        
        par30_index = individual_latent_factors < scinorm.ppf(counterpary.PD)
        recovery_index = individual_latent_factors > scinorm.ppf(1-recovery_prob)
        
        
        for counterparty, individual_latent_factor in zip(counterparties, individual_latent_factors):
            c_beta = beta.ppf(individual_latent_factor, counterparty.alpha, counterparty.beta)