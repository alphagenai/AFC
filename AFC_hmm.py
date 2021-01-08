# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:12:37 2021

@author: mark
"""

from hmmlearn import hmm
# Setting the HMM structure. n_component is the number of hidden states
model = hmm.MultinomialHMM(n_components=2)



# Training the model with your data
one_contract_id = '1352353'
data = ~no_elec.loc[:, one_contract_id]
data = data.fillna(0).astype('int').values.reshape(-1,5)
model.fit(data ) Z = model.predict(data)
# Predicting the states for the observation sequence X (with Viterbi)
