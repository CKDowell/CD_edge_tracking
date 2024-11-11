# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:34:25 2024

@author: dowel
"""

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
#%%
plt.close('all')
meta_data = {'stim_type': 'plume',
              'act_inhib':'act',
    'ledOny': -float(50)/2,
              'ledOffy':float(50)/2,
              'ledOnx': 10,
              'ledOffx': 30,
              'LEDoutplume': False,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 0,
              'RepeatInterval':250
              }

rootdir = "Y:\Data\Optogenetics\\37G12_PFR_a\\37G12_PFR_a_inhibition\\Test"
#
flies = [
     #"241013\\f1\\Trial1",
#     "241013\\f2\\Trial1",
#     "241013\\f3\\Trial1",
#     "241014\\f1\\Trial1",
#     "241014\\f2\\Trial1",
#     "241014\\f3\\Trial1",
#     "241014\\f4\\Trial1",
#     "241014\\f5\\Trial1",
#     "241015\\f1\\Trial1",
#     "241015\\f2\\Trial1",
#     "241015\\f3\\Trial1",
#     "241015\\f4\\Trial1",
#     "241015\\f5\\Trial1",
#     "241015\\f6\\Trial1",
#     "241015\\f4\\Trial2",
#     "241015\\f5\\Trial2",
#     "241015\\f6\\Trial2",
    
    "241023\\f1\\Trial1",#0 thresh
    "241023\\f2\\Trial1",# crap walker
    "241015\\f3\\Trial1",# 0 thresh
    "241015\\f4\\Trial1",# led off
    "241015\\f5\\Trial1",# crap walker
    
    "241107\\f1\\Trial1",# 0 thresh
    "241107\\f3\\Trial1",# led off
    "241107\\f4\\Trial1"# -1000 mm threshold
    
    
        ]

for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)