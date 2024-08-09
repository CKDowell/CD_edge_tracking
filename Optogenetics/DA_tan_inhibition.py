# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:02:21 2024

@author: dowel

Inhibition of DA tangential neurons innervating upper fan-shaped body layers
FB5H, FB6H, FB7B

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
              'act_inhib':'inhib',
    'ledOny': 700,
              'ledOffy':'all',
              'ledOnx': -1000,
              'ledOffx': 1000,
              'LEDoutplume': True,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 0,
            
              }
rootdir = 'Y:\Data\Optogenetics\DA_tan_SS56699\DA_Tan_inhib'
flies = [
    "240723\\f3\\Trial1",
    #240723\\f6\\Trial1",
    #"240723\\f9\\Trial1",
    #"240724\\f4\\Trial1",
    "240730\\f3\\Trial1"]

plt.rcParams['pdf.fonttype'] = 42 
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)