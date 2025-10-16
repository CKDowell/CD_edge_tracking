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
    
    
#%% Activation Pulses
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
#%% Inhibition alternation
rootdir = r"Y:\Data\Optogenetics\FB4P_b_SS60296\FB4P_b_SS60296_Corridor_Outside_Stim_Inhib\Test"
plt.close('all')
flies = [
   # r'250121\f1\Trial1', Did not make stim
      #   r'250121\f2\Trial1', Did not make stim
         #r'250121\f4\Trial1', Did not make stim
         r'250121\f6\Trial1',
         
         r'250122\f1\Trial1',
         #r'250122\f3\Trial1', Did not make stim
         #r'250122\f5\Trial1', Did not make stim
         #r'250122\f6\Trial1', Did not make stim
         
         r'250124\f1\Trial1',
         r'250124\f3\Trial1',
        # r'250124\f5\Trial1', Did not make stim
         
        # r'250127\f1\Trial1', Did not make stim
         #r'250127\f2\Trial1',
         #r'250127\f5\Trial1',# Did not make stim
         #r'250127\f6\Trial1',
         
         #r'250128\f1\Trial1',
         r'250128\f3\Trial1',
         ]
meta_data['stim_type'] = 'alternation'
meta_data['act_inhib'] = 'inhib'
for i,f in enumerate(flies):
    plt.figure()
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
#%% Sansu test
rootdir = r"F:\MB022B_GtACR1"
plt.close('all')
flies = [
    
    r'250911\f1\Trial1_Control',
    r'250911\f1\Trial2_Test',
    
    r'250912\f1\Trial1_Control',
    r'250912\f1\Trial2_Test',
    
    r'250919\f1\Trial1_Control',
    r'250919\f1\Trial3_Test',
    
    r'250916\f1\Trial1_Test',
    r'250916\f1\Trial2_Control',
    
    
    
    r'250919\f3\Trial1_Test',
    r'250919\f3\Trial2_Control',
    
    r'250923\f1\Trial1_Test',
    r'250923\f1\Trial2_Control',
    ]

meta_data['stim_type'] = 'alternation_jump'
meta_data['act_inhib'] = 'inhib'
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    tfile = [ir for ir in files if ir.endswith('.log')][0]
    savepath = os.path.join(datadir,tfile)
    df = fc.read_log(savepath)
    op = opto()
    plt.figure(i)
    try:
        meta_data['stim_type'] = 'alternation_jump'
        op.plot_plume_simple(meta_data,df)
    except:
        meta_data['stim_type'] = 'alternation'
        op.plot_plume_simple(meta_data,df)
    plt.title(f)

#%% Inhibition jumps - Test
# Need: 4 more good trackers before analysis
rootdir = r"Y:\Data\Optogenetics\FB4P_b_SS60296\FB4P_b_SS60296_Inhibition_Jumps"
plt.close('all')
flies = [
    r'Test\250212\f1\Trial1', #Good tracker
    r'Test\250212\f3\Trial1', #One jump
    r'Test\250218\f1\Trial1', # Good tracker
    r'Test\250218\f3\Trial1', # Good tracker
    #r'Test\250219\f1\Trial1',
    #r'Test\250220\f1\Trial1',
    r'Test\250220\f3\Trial1',# One jump
    r'Test\250221\f1\Trial1' # Good tracker
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
    try:
        meta_data['stim_type'] = 'alternation_jump'
        op.plot_plume_simple(meta_data,df)
    except:
        meta_data['stim_type'] = 'alternation'
        op.plot_plume_simple(meta_data,df)


#%% Inhibition jumps - Control +/GtACR1
rootdir = r"Y:\Data\Optogenetics\FB4P_b_SS60296\FB4P_b_SS60296_Inhibition_Jumps"
plt.close('all')
# Need: 4 more good trackers before analysis
flies =[
        r'Control\250212\f2\Trial1',# Good tracker
        r'Control\250212\f4\Trial1',# Good tracker
        r'Control\250218\f2\Trial1',# Good tracker
        r'Control\250220\f2\Trial1',# Good tracker
        r'Control\250221\f2\Trial1',# Strange downwind animal
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
    try:
        meta_data['stim_type'] = 'alternation_jump'
        op.plot_plume_simple(meta_data,df)
    except:
        meta_data['stim_type'] = 'alternation'
        op.plot_plume_simple(meta_data,df)









