# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:27:40 2020

@author: mark
"""

import pandas as pd
import numpy as np

from scipy.stats import binom

from calc_PD import calc_PD
from monthly_averages import calc_moving_average

df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)

df = df.groupby(['ContractId', 'TransactionTS']).sum()

daily_sdf = df.groupby(['ContractId', pd.Grouper(freq='1D', level=1)]).sum()
monthly_sdf = daily_sdf.groupby(['ContractId',pd.Grouper(freq='M', level=1)])['AmountPaid'].sum()
monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)


PD_given_D, PD_given_ND, PND_given_D, PND_given_ND = calc_PD(monthly_sdf_pivot,)
monthly_sdf_fullts = calc_moving_average(monthly_sdf)

ma_pivot = monthly_sdf_fullts['MovingAverage'].unstack(0).shift(1).fillna(method='ffill') #shift(1) for next month forecast, ffill for future months with no MA (because no payments made)

## logic

"""

if month = paid:
    month+1 = paid*PNDgivenND + PDgivenND*0
elif month != paid
    month+1 = paid*PNDgivenD + PDgivenD*0
    
"""


one_month_forecast_if_paid = ma_pivot.multiply(PND_given_ND)
one_month_forecast_if_default = ma_pivot.multiply(PND_given_D)


one_contract_id = monthly_sdf_pivot.columns[3]

one_ma = ma_pivot[one_contract_id]
forecast_date = '2019-12-31'

#all_dates = monthly_sdf_pivot.index.get_level_values(0)[forecast_date:]

defaults = monthly_sdf_pivot.iloc[1:]==0 # total non-NaN: 28,593
t=0
if defaults.loc[forecast_date, one_contract_id]: # paid
    EV_ND[t+1] = one_ma[forecast_date]*PND_given_ND
else: # not paid
    EV_ND[t+1] = one_ma[forecast_date]*PND_given_D


## don't exactly add to 1
PD = PD_given_D + PD_given_ND
PND = PND_given_D + PND_given_ND

PND_given_ND*PND_given_ND

PND_given_ND*PD_given_ND

PD_given_ND*PND_given_D

PD_given_ND*PD_given_D





average_payment = one_ma[forecast_date]
start_date_payment = defaults.loc[forecast_date, one_contract_id] #bool

if start_date_payment:
    node1 = Node(1, average_payment, None)
    node1.p = PND_given_ND

    node2 = Node(1, 0, None)
    node2.p = PD_given_ND

else:
    node3 = Node(1, average_payment, None)
    node3.p = PND_given_D

    node4 = Node(1, 0)
    node4.p = PD_given_D
    
node5 = Node(2, average_payment, node1)
node5.p = node1.p*PND_given_ND


node6 = Node(2, 0, node1)

node7 = Node(2, average_payment, node2)

node8 = Node(2, 0, node2)

class Node(object):
    def __init__(self, t, val, prev_node):
        self.t = t
        self.value = val
        if prev_node is not None:
            if prev_node.value & val:
                self.p = prev_node.p*PND_given_ND
            elif prev_node.value & ~val:
                self.p = prev_node.p*PD_given_ND
            elif ~prev_node.value & val:
                self.p = prev_node.p*PND_given_D
            elif ~prev_node.value & ~val:
                self.p = prev_node.p*PD_given_D
            else:
                raise ValueError('Something went wrong')
        
class LatticeModel(object):
    def __init__(self, initial_payment, number_of_timepoints, average_payment, 
                 contract_id):
        self.initial_payment = initial_payment
        self.number_of_timepoints = number_of_timepoints
        self.average_payment = average_payment
        self.contract_id = contract_id
        

        if initial_payment:
            self.initial_node = Node(0, initial_payment, None)  
            self.initial_node.p = 1
        else:
            self.initial_node = Node(0, 0, None)
            self.initial_node.p = 1

        self.nodes_dict = {0:initial_node}            

    def add_level(self, t):
        """ create a new set of nodes for the next timepoint """
        new_nodes = []
        for node in self.nodes_dict[t]:
            node.offspring1 = Node(t+1, self.average_payment, node)
            node.offspring2 = Node(t+1, 0, node)
            new_nodes.append([node.offspring1, node.offspring2])
        self.nodes_dict[t+1] = new_nodes