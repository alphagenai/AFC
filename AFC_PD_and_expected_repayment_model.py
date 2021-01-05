# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:27:40 2020

@author: mark
"""

import pandas as pd
import numpy as np
import seaborn as sns

from scipy.stats import binom
from matplotlib import pyplot as plt

from calc_PD import calc_PD
from monthly_averages import calc_moving_average




class Node(object):
    def __init__(self, t, val, prev_node, PD_dict):
        self.t = t
        self.value = val
        if prev_node is not None:
            self.cum_val = prev_node.cum_val+val

            prev_paid = (prev_node.value != 0)
            paid = (val != 0)
            if prev_paid & paid:
                self.p = prev_node.p*PD_dict['PND_given_ND']
            elif prev_paid & ~paid:
                self.p = prev_node.p*PD_dict['PD_given_ND']
            elif ~prev_paid & paid:
                self.p = prev_node.p*PD_dict['PND_given_D']
            elif ~prev_paid & ~paid:
                self.p = prev_node.p*PD_dict['PD_given_D']
            else:
                raise ValueError('Something went wrong')
        else:
            self.cum_val = val



def weighted_kde(x, weights, **kwargs):
    sns.kdeplot(x, weights=weights, **kwargs)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


class LatticeModel(object):
    """ TO DO: ENSURE TOTAL PAYMENTS DO NOT GO OVER 100% """
    def __init__(self, initial_payment, average_payment, 
                 contract_id, PD_dict):
        self.initial_payment = initial_payment
        self.average_payment = average_payment
        self.contract_id = contract_id
        self.PD_dict=PD_dict

        if initial_payment:
            initial_node = Node(0, initial_payment, None, PD_dict)  
            initial_node.p = 1
        else:
            initial_node = Node(0, 0, None, PD_dict)
            initial_node.p = 1

        self.nodes_dict = {0:[initial_node,]}            

    def add_level(self,):
        """ create a new set of nodes for the next timepoint """
        t = max(self.nodes_dict.keys())
        new_nodes = []
        for node in self.nodes_dict[t]:
            node.offspring1 = Node(t+1, self.average_payment, node, self.PD_dict)
            node.offspring2 = Node(t+1, 0, node, self.PD_dict)
            new_nodes.extend([node.offspring1, node.offspring2])
        self.nodes_dict[t+1] = new_nodes

    def plot_forecasts(self, T):
        t_list = [node.t for tau in range(T) for node in self.nodes_dict[tau]]
        p_list = [node.p for tau in range(T) for node in self.nodes_dict[tau]]
        v_list = [node.cum_val for tau in range(T) for node in self.nodes_dict[tau]]
        
        df = pd.DataFrame(dict(t=t_list, v=v_list, p=p_list))
        self._df = df = df.groupby(['t','v']).sum().reset_index()

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        
        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(df, row="t", hue="t", aspect=15, height=.5, palette=pal)
        
            
        # Draw the densities in a few steps
        g.map(weighted_kde, "v", "p",
              bw_adjust=.5, 
              clip_on=False,
              fill=True, 
              alpha=1, linewidth=1.5)
        g.map(weighted_kde, "v", "p", 
              clip_on=False, color="w", lw=2, bw_adjust=.5)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)
        

        g.map(label, "v")
        
        # Set the subplots to overlap
        g.fig.subplots_adjust(hspace=-.25)
        
        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)        
        
        
if __name__ == "__main__":


    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    
    monthly_sdf = df.groupby(['ContractId',pd.Grouper(key='TransactionTS', freq='M',)])['AmountPaid'].sum()
    monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
    
    
    PD_dict = calc_PD(monthly_sdf_pivot,)
    monthly_sdf_fullts = calc_moving_average(monthly_sdf)
    
    ma_pivot = monthly_sdf_fullts['MovingAverage'].unstack(0).shift(1).fillna(method='ffill') #shift(1) for next month forecast, ffill for future months with no MA (because no payments made)
    

    one_contract_id = monthly_sdf_pivot.columns[3]
    one_ma = ma_pivot[one_contract_id]
    forecast_startdate = '2019-12-31'
    
    forecast_dates = monthly_sdf_pivot[forecast_startdate:].index

    
    average_payment = one_ma[forecast_startdate]
    initial_payment = monthly_sdf_pivot.loc[forecast_startdate, one_contract_id] #bool


    lm = LatticeModel(initial_payment=0, average_payment=average_payment, 
                      contract_id=one_contract_id, PD_dict=PD_dict)
    
    for t in range(7):
        lm.add_level()
    
    lm.plot_forecasts(6)
    print(lm._df)