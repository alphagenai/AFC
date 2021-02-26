# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:40:34 2021

@author: mark
"""


import pandas as pd
import numpy as np

from basic_datasets import BasicDatasets

bd = BasicDatasets()
dts = bd.daily_ts

dts['loss'] = 1-dts['cum_val'] 
ead = dts[dts['PAR30+']]['loss']

check = dts[dts['PAR30+']]

dts['PAR30+'].diff().fillna(False).loc['1349704', '2020-09-30']

dts['seq_no'] = dts['PAR30+'].diff().cumsum()

dts.loc['1362450'].to_csv('temp5.csv')

spd = dts[dts['PAR30+']].reset_index().set_index('seq_no')

default_data = spd.groupby('seq_no').min().rename(columns={'TransactionTS':'seq_start', 
                                            'loss':'min_loss'}).join(
    spd.groupby('seq_no').max()[['TransactionTS', 'loss', 'days_out_of_elec']].rename(columns={
        'TransactionTS':'seq_end', 'loss':'max_loss', 'days_out_of_elec':'max_days_out_of_elec'})
    ).join(spd[spd['days_out_of_elec']==31][['loss']].rename(columns={'loss':'EAD'}))

