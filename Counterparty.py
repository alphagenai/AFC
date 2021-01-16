# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:05:43 2021

@author: mark
"""


class Counterparty(object):
    def __init__(self, ContractId):
        self.ContractId = ContractId
        self.alpha = 1
        self.beta = 1
        self.PD = 0.055
        self.RSQ = 0.3
        
    def percent_timeseries(self, startdate=None, enddate=None):
        raise NotImplementedError()
        
    