# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:52:09 2020

@author: mark
"""
# Import pyplot, figures inline, set style, plot pairplot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.pairplot(all_features, hue='Age');