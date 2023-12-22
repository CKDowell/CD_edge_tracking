# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:28:47 2023

@author: dowel
"""
#%% 
import numpy as np
import pandas as pd
import pickle as pkl
from analysis_funs.regression import fci_regmodel
import matplotlib.pyplot as plt
#%% 
filename = "Y:\Data\FCI\AndyData\KC_dLight\\all_data\\all_data.pkl"
data = pd.read_pickle(filename)
#%% 
dlist = data.keys()
dlist_n = np.empty(0)
for d in dlist:
    dlist_n = np.append(dlist_n,d)
dnum = 0
t_dat = data[dlist_n[0]]

#%%
t_dat2 = t_dat['a1']
y = t_dat2['G5']
ft2 = t_dat2
pv2 = t_dat2
regchoice = ['odour onset', 'odour offset','cos heading pos']

# , 'in odour', 
#                                 'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
#                                 'angular velocity pos','x pos','x neg','y pos', 'y neg','ramp down since exit','ramp to entry'] 


fc = fci_regmodel(y,ft2,pv2)
fc.run(regchoice)
fc.run_dR2(20,fc.xft)
fc.plot_flur_w_regressors(regchoice)
fc.plot_example_flur()