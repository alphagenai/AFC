# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:27:40 2020

@author: mark
"""

import pandas as pd
import numpy as np
import seaborn as sns
import logging

from scipy.stats import binom
from matplotlib import pyplot as plt
from math import isclose

from calc_PD import PDCalculator
from monthly_averages import calc_moving_average
from individual_analysis1 import create_percent_sdf

""" TO DO: CHECK THAT THE CUMULATIVE VALUES ARE CORRECT """


def initialise(use_percent):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    df = pd.read_pickle('files\\small_df_1000_dec_17.pkl')
    df['TransactionTS'] = pd.to_datetime(df['TransactionTS'],format='%Y/%m/%d %H:%M:%S').dt.tz_localize(None)
    
    forecast_startdate = '2019-6-30'

    monthly_sdf = df.groupby(['ContractId',pd.Grouper(key='TransactionTS', freq='M',)])['AmountPaid'].sum()
        

    if use_percent:
        monthly_sdf_pivot, monthly_averages = initialise_percent(monthly_sdf)
    else:
        monthly_sdf_pivot, monthly_averages = initialise_values(monthly_sdf)

    pd_calc = PDCalculator(monthly_sdf_pivot)
    _, _ = pd_calc.data_prep()

    ma_pivot = monthly_averages['MovingAverage'].unstack(0).shift(1) #shift(1) for next month forecast

    one_contract_id = monthly_sdf_pivot.columns[3]  # original example
    one_contract_id = monthly_sdf_pivot.columns[8]  # someone new
    one_contract_id = '1349968'


    one_ma = ma_pivot[one_contract_id]
    if one_ma.isna().any():
        logging.warning("Not enough payment data to calculate moving average")
    
    #forecast_dates = monthly_sdf_pivot[forecast_startdate:].index

    counterparty = Counterparty(one_contract_id)
    counterparty.update_bayesian_mean_PDs(monthly_sdf_pivot, forecast_startdate, pd_calc)

    average_payment = one_ma[forecast_startdate]
    if np.isnan(average_payment):
        logging.warning("Not enough payment data to calculate moving average")
        average_payment = monthly_sdf_pivot.loc[:forecast_startdate, one_contract_id].mean()

    initial_payment = monthly_sdf_pivot.loc[forecast_startdate, one_contract_id] 
    paid_so_far = monthly_sdf_pivot.loc[:forecast_startdate, one_contract_id].sum()
        
    return initial_payment, paid_so_far, average_payment, counterparty, forecast_startdate, monthly_sdf_pivot

def initialise_percent(monthly_sdf):
    monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
    monthly_perc_sdf_pivot = create_percent_sdf(monthly_sdf_pivot,
                                                cumulative=False, use_monthdiff=False, cohort='dec_17')

    monthly_averages = calc_moving_average(monthly_perc_sdf_pivot.stack().to_frame('AmountPaid'))
    
    return monthly_perc_sdf_pivot, monthly_averages

def initialise_values(monthly_sdf):
    monthly_sdf_pivot = monthly_sdf.unstack(0).fillna(0)
    monthly_averages = calc_moving_average(monthly_sdf.to_frame())

    return monthly_sdf_pivot, monthly_averages
    


class Node(object):
    def __init__(self, t, val, prev_node, PD_dict, use_percent_val):
        self.t = t
        self.value = val
        if prev_node is not None:
            if use_percent_val:
                if prev_node.cum_val == 1.0:
                    self.cum_val = 1.0
                    self.p = prev_node.p
                    return
                else:
                    self.cum_val = np.min([prev_node.cum_val+val, 1.0])  #nodes cannot be worth more than 100%
            else:
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
            if use_percent_val:
                self.cum_val = np.min([1.0, val])
            else:
                self.cum_val = val
                
    def __repr__(self):
        return '<Node[{}]: value: {:.2f}, prob:{:.2f}, cum_val: {:.2f}>'.format(self.t,
                                                                 self.value,
                                                                 self.p,
                                                                 self.cum_val,
                                                                 )




def weighted_kde(x, weights, **kwargs):
    sns.kdeplot(x, weights=weights, **kwargs)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


class LatticeModel(object):
    """ TO DO: ENSURE TOTAL PAYMENTS DO NOT GO OVER 100% """
    def __init__(self, initial_payment, paid_so_far, average_payment, 
                 contract_id, forecast_startdate, PD_dict, use_percent=True):
        self.initial_payment = initial_payment
        self.average_payment = average_payment
        self.contract_id = contract_id
        self.forecast_startdate = forecast_startdate
        self.PD_dict=PD_dict
        self.use_percent = use_percent

        if initial_payment:
            initial_node = Node(0, initial_payment, None, PD_dict, self.use_percent)  
        else:
            initial_node = Node(0, 0.0, None, PD_dict, self.use_percent)
        initial_node.p = 1
        initial_node.cum_val = paid_so_far
        logging.debug('Node created: {}'.format(repr(initial_node)))

        self.nodes_dict = {0:[initial_node,]}            

    def add_level(self,):
        """ 
            create a new set of nodes for the next timepoint 
            TO DO: Engineering problem of how to combine nodes with same value
        """
        t = max(self.nodes_dict.keys())
        new_nodes = []
        for node in self.nodes_dict[t]:
            node.offspring1 = Node(t+1, self.average_payment, node, self.PD_dict, self.use_percent)
            logging.debug('Node created: {}'.format(repr(node.offspring1)))
            new_nodes.extend([node.offspring1,])
            
            if (self.use_percent and node.cum_val == 1.0):
                pass # we only want one child node if we already have paid off the contract
            else: # contract not paid off
                node.offspring2 = Node(t+1, 0.0, node, self.PD_dict, self.use_percent)
                logging.debug('Node created: {}'.format(repr(node.offspring2)))

                new_nodes.extend([node.offspring2,])
                
        self.nodes_dict[t+1] = new_nodes
        self.check_probabilities(new_nodes)

    def check_probabilities(self, nodes):
        p=0.0
        for node in nodes:
            p+=node.p
        assert isclose(p, 1.0, abs_tol=1e-8), 'probabilities do not sum to 1.0 for counterparty {}, sum is {}'.format(self.contract_id, p)
        
    def calculate_expectation(self, t):
        probs = np.array([node.p for node in self.nodes_dict[t]])
        vals = np.array([node.cum_val for node in self.nodes_dict[t]])
        return np.dot(probs, vals)
        
    def plot_forecasts(self, T):
        t_list = [node.t for tau in range(T) for node in self.nodes_dict[tau]]
        p_list = [node.p for tau in range(T) for node in self.nodes_dict[tau]]
        v_list = [node.cum_val for tau in range(T) for node in self.nodes_dict[tau]]
        
        df = pd.DataFrame(dict(t=t_list, v=v_list, p=p_list))
        self._df = df = df.groupby(['t','v']).sum().reset_index()

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        
        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(df, row="t", hue="t", aspect=15, height=.5, palette=pal,
                          row_order=df['t'].unique()[::-1])
        
            
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
        g.axes[-1][0].set_xlabel('Cumulative Value')
        g.axes[3][0].set_ylabel('Time')
        title = 'forecast for {} at {}'.format(self.contract_id, self.forecast_startdate)
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title)
        plt.savefig('files\\'+title)
        
        
if __name__ == "__main__":
    from Counterparty import Counterparty
    
    use_percent=True
    
    initial_payment, paid_so_far, average_payment, counterparty, forecast_startdate, monthly_sdf_pivot = initialise(use_percent)

    lm = LatticeModel(initial_payment, paid_so_far=paid_so_far, average_payment=average_payment, 
                      contract_id=counterparty.ContractId, forecast_startdate=forecast_startdate, 
                      PD_dict=counterparty.PD_dict, use_percent=use_percent)
    
    for t in range(7):
        lm.add_level()
    
    lm.plot_forecasts(6)
    print(lm._df)
    
    fig, ax = plt.subplots()
    title = 'Realised Cumulative Payments'
    actual = paid_so_far + monthly_sdf_pivot.loc[
        forecast_startdate:pd.Timestamp(forecast_startdate)+pd.DateOffset(months=5), counterparty.ContractId
                                   ].cumsum()
    ax = actual.plot(ax=ax,title = title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Value')
    plt.title(title)
    plt.savefig('files\\'+title)
        
    
    
# def plot_forecasts(tss, forecasts, past_length, num_plots):
#     for target, forecast in islice(zip(tss, forecasts), num_plots):
#         ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
#         forecast.plot(color='g')
#         plt.grid(which='both')
#         plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
#         plt.show()
