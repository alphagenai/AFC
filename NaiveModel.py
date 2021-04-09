# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:02:31 2021

@author: mark
"""


import pandas as pd

from matplotlib import pyplot as plt

from calculate_returns import RegModelDatasets

class NaiveModel(object):
    def __init__(self, rmds, month):
        self.rmds = rmds
        self.month = month
        
    def hist_rate(self, num_months=6):
        self.num_months = num_months
        hist_df = self.rmds.mpp.loc[:self.month].iloc[-num_months:]
        r = hist_df.sum() / hist_df.index.size
        self._hr = r
        return r

    def forecast_rate(self, ):
        num_months = self.rmds.mpp.loc[self.month:].index.size - 2 #we exclude current month and '2020-11-30'
        fr = num_months*self.hist_rate()
        self._fr = fr
        return fr

    def forecast_loss(self):
        return 1 - (self.rmds.mcpp.loc[self.month] + self.forecast_rate()).clip(0,1)
    
    def error(self):
        return rmds.final_loss - self.forecast_loss()
    
    def plot_errors(self,):
        fig, ax = plt.subplots()
        ax.hist(self.error(), bins=50)
        plt.title(self.month.date())
        plt.savefig('files\\naive model errors {}'.format(self.month.date()))
        plt.close()
        
    def debug(self):
        pd.concat([rmds.final_loss , self.forecast_loss(), self._fr]).to_csv('files\\debug naive model {}'.format(self.month.date()))
            

if __name__ == "__main__":
    rmds = RegModelDatasets()
    month = '2020-09-30'
    nm_dict = {}
    for month in rmds.mcpp.index:
        nm=NaiveModel(rmds, month)
        nm_dict[month] = nm
        nm.plot_errors()
        nm.debug()