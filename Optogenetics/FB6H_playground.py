# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:28:46 2025

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
#%% Blanket inhibition - Test
plt.close('all')
rootdir = r'Y:\Data\Optogenetics\FB6H_SS95649\FB6H_inhibition'
flies = [r'Test\250213\f1\Trial1',# Good tracker, walked off
         #r'Test\250214\f1\Trial1', # Did not make inhib
         r'Test\250214\f3\Trial1',# Decent tracker, no obvious phen
         r'Test\250214\f5\Trial1',# Decent tracker, long returns
         r'Test\250218\f1\Trial1', # Strange tracker - not ET
         r'Test\250225\f1\Trial1', # Spiraling after stimulation, poor returns
         r'Test\250225\f3\Trial1', # Walked off
         r'Test\250227\f1\Trial1', # Walked off, plume cross overs
         #r'Test\250227\f3\Trial1', # Did not make inhib
         r'Test\250227\f5\Trial1', # Very long returns, probably not an ET
         r'Test\250228\f1\Trial1', # Good tracker then affected
         #r'Test\250228\f3\Trial1',
         #r'Test\250228\f5\Trial1',
         r'Test\250304\f1\Trial1', # ET, hard to tell if there is an effect
         r'Test\250304\f4\Trial1',# Walked off
         r'Test\250305\f1\Trial1', # Long returns
         r'Test\250306\f2\Trial1', # Normal ET
         r'Test\250306\f5\Trial1',# Walked off
         ]
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
    plt.title(f)
    
#%% Blanket inhibition - Control
plt.close('all')
rootdir = r'Y:\Data\Optogenetics\FB6H_SS95649\FB6H_inhibition'
flies = [
    #r'Control\250213\f2\Trial1',
        # r'Control\250214\f2\Trial1',
         r'Control\250214\f4\Trial1', # Cross over not ET
         #r'Control\250218\f2\Trial1',
         r'Control\250225\f2\Trial1', #Good tracker
         #r'Control\250227\f2\Trial1',
         r'Control\250227\f4\Trial1',# Not ET
         r'Control\250228\f2\Trial1',#Not ET
         r'Control\250228\f4\Trial1',# ET
        # r'Control\250304\f2\Trial1',
        # r'Control\250304\f3\Trial1',
        # r'Control\250304\f5\Trial1',
        # r'Control\250304\f6\Trial1',
         r'Control\250305\f1\Trial1',# Experiment not run for long enough
         r'Control\250305\f2\Trial1',# ET, some change after light
        # r'Control\250306\f1\Trial1',
         r'Control\250306\f3\Trial1',# Experiment not run for long enough
         r'Control\250306\f4\Trial1',#No ET before light, ET after light
         #r'Control\250306\f6\Trial1',
        # r'Control\250306\f7\Trial1',
         #r'Control\250306\f8\Trial1',
         ]

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
    plt.title(f)








