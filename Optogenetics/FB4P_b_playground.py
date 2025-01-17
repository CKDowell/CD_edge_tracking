# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:30:53 2024

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
              'ledOny': 600,
              'ledOffy':'all',
              'ledOnx': -1000,
              'ledOffx': 1000,
              'LEDoutplume': True,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 0,
            
              }
#%% Visualise data - test animals
rootdir = 'Y:\Data\Optogenetics\FB4P_b_SS60296\FB4P_b_SS60296_Chrimson_Corridor_outside\Test_Flies'
plt.close('all')
flies = [
    '240823\\f1\\Trial1',
    '240823\\f2\\Trial2',
    '240823\\f3\\Trial2',
    '240827\\f3\\Trial1']

for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
#%% Stim alternation
rootdir = 'Y:\Data\Optogenetics\FB4P_b_SS60296\FB4P_b_SS60296_Chrimson_Corridor_outside_alt\Test_Flies'
plt.close('all')
flies = [
    '240827\\f3\\Trial1']
meta_data['stim_type'] = 'alternation'
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
#%% Pulses
rootdir = 'Y:\Data\Optogenetics\FB4P_b_SS60296\FB4P_b_SS60296_Chrimson_Pulses\Test_Flies'
plt.close('all')
flies = [
    '240823\\f1\\Trial1',
    '240823\\f2\\Trial1',
    '240823\\f3\\Trial1',
    '240827\\f3\\Trial1']
meta_data['stim_type'] = 'pulse'
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)