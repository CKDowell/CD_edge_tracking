# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:28:59 2025

@author: dowel
"""

import numpy as np
import pandas as pd
import analysis_funs.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
import pickle 
meta_data = {'stim_type': 'alternation',
              'act_inhib':'act',
              'ledOny': 700,
              'ledOffy':'all',
              'ledOnx': -1000,
              'ledOffx': 1000,
              'LEDoutplume': True,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 0,
            
              }
#%% Entry count alt - inhibition
plt.close('all')
rootdir = r'Y:\Data\Optogenetics\FB5I_SS100553\FB5I_inhibition\Test'

flies = [r'250311\f1\Trial1',
         r'250311\f2\Trial1',
         r'250312\f1\Trial1',
         r'250312\f3\Trial1',
        # r'250312\f5\Trial1',
]
meta_data['stim_type'] = 'alternation_jump'
meta_data['act_inhib'] = 'inhib'

for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)


#%% Entry conut alt - activation
plt.close('all')
rootdir = r'Y:\Data\Optogenetics\FB5I_SS100553\FB5I_activation\Test'

flies = [r'250312\f4\Trial1',
         r'250312\f5\Trial1',
]
meta_data['stim_type'] = 'alternation_jump'
meta_data['act_inhib'] = 'act'

for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
