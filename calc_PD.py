# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 00:12:12 2020

@author: mark
"""



default = daily_sdf_fullts['PAR30+']
default_event = (default.diff() != 0) & (daily_sdf_fullts['PAR30+']==True)
undefault_event = (default.diff() != 0) & (daily_sdf_fullts['PAR30+']==False) & (
    (default.index.get_level_values(1).month != 12) & (default.index.get_level_values(1).year != 2017)
    )
daily_sdf_fullts[undefault_event].to_csv('undefaults.csv')
daily_sdf_fullts[default_event].to_csv('defauls.csv')