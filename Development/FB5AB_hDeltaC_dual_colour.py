# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:23:48 2026

@author: dowel
"""

from analysis_funs.regression import fci_regmodel

import numpy as np
import pandas as pd
import analysis_funs.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from analysis_funs.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from scipy import stats
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
from analysis_funs.utilities import funcs as fn
from Utilities.utils_general import utils_general as ug
from Utilities.utils_plotting import uplt as uplt
plt.rcParams['pdf.fonttype'] = 42 
#%%

datadirs = [
    #r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260202\f2\Trial1', # missing .dat file
            #r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260202\f2\Trial2',
            #r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260202\f2\Trial3',
            #r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260202\f2\Trial4',#Octanol pulses
            #r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260202\f2\Trial5',#ACV pulses
            
            r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260203\f1\Trial1',
            r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260203\f1\Trial2',# some ET
            r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260203\f1\Trial3',#oct pulses
            r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260203\f1\Trial4',#ACV pulses
            
            
            
            
            ]
regions = ['fsb2','fsbTN1']
for datadir in datadirs:
   # regions = ['eb','fsb_upper_1','fsb_lower_1','fsb_upper_2']
    
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    
    cx = CX(name,regions,datadir)
    
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois()
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()#upsample to 50Hz
    pv2, ft, ft2, ix = cx.load_postprocessing()
    #Channel 2 = Green, Channel 1 = red
    regions2 = ['fsb2_ch2','fsb2_ch1']
    cxa = CX_a(datadir,regions=regions2,yoking=True)
    cxa.save_phases()
    
    
#%% Check data
regions2 = ['fsbtn1_ch1','fsb2_ch2']

datadir = r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260203\f1\Trial4'
cxa = CX_a(datadir,regions=regions2,yoking=False,denovo=False)

#%% 
fb5ab = cxa.pv2['0_fsbtn1_ch1'].to_numpy()
hdc = cxa.pv2['0_fsbtn1_ch2'].to_numpy()
#hdc2 = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
t = cxa.pv2['relative_time'].to_numpy()

plt.plot(t,fb5ab,color='k')
#plt.plot(t,-fb5ab+1,color='m')
plt.plot(t,hdc,color='b')
#plt.plot(t,hdc2,color='b')
ins = cxa.ft2['instrip'].to_numpy().astype(float)

if np.sum(ins)>0:
    #plt.plot(t,ins,color='r')
    plt.fill_between(t,ins*0,ins*1.5,color=[1,.2,.2],linewidth=0)
else:
    ins = cxa.ft2['mfc3_stpt'].to_numpy()>0
    plt.plot(t,ins,color='g')
plt.ylabel('dF/F0')
plt.xlabel('time (s)')
#%% Edge tracking  recovery
datadirs = [r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260203\f1\Trial2']
meandat = np.zeros((600,2,len(datadirs)))
for di,d in enumerate(datadirs):
    cxa = CX_a(d,regions=regions2,yoking=False,denovo=False)
    fb5ab = cxa.pv2['0_fsbtn1_ch1'].to_numpy()
    #hdc = cxa.pv2['0_fsbtn1_ch2'].to_numpy()
    hdc = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
    jumps = cxa.get_entries_exits_like_jumps()
    all_js = np.zeros((600,2,len(jumps)))
    for i,j in enumerate(jumps):
        dx = np.arange(j[1],j[2])
       # plt.plot(amp[dx],color='k',alpha=0.3)
        tamp = fb5ab[dx]
        amplen  = np.min([600,len(tamp)])
        all_js[:amplen,0,i] = tamp[:amplen]
        
        tamp = hdc[dx]
        amplen  = np.min([600,len(tamp)])
        all_js[:amplen,1,i] = tamp[:amplen]
        
    all_js[all_js==0] = np.nan    
    meandat[:,:,di] = np.nanmean(all_js,axis=2)
colours = ['r','m']
for i in range(2):
    plt.plot(meandat[:,i],color=colours[i])

#%% ACV pulses recovery
datadirs = [r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260202\f2\Trial5',
    r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260203\f1\Trial4']
meandat = np.zeros((600,2,len(datadirs)))
x = np.arange(0,600)/10

for di,d in enumerate(datadirs):
    cxa = CX_a(d,regions=regions2,yoking=False,denovo=False)
    fb5ab = cxa.pv2['0_fsbtn1_ch1'].to_numpy()
    #hdc = cxa.pv2['0_fsbtn1_ch2'].to_numpy()
    hdc = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
    jumps = cxa.get_entries_exits_like_jumps()
    all_js = np.zeros((600,2,len(jumps)))
    for i,j in enumerate(jumps):
        dx = np.arange(j[0],j[2])
       # plt.plot(amp[dx],color='k',alpha=0.3)
        tamp = fb5ab[dx]
        amplen  = np.min([600,len(tamp)])
        all_js[:amplen,0,i] = tamp[:amplen]
        
        tamp = hdc[dx]
        amplen  = np.min([600,len(tamp)])
        all_js[:amplen,1,i] = tamp[:amplen]
        
    all_js[all_js==0] = np.nan    
    meandat[:,:,di] = np.nanmean(all_js,axis=2)
colours = ['r','m']
for i in range(2):
    plt.plot(x,meandat[:,i],color=colours[i])
    

#%% Octanol pulses recovery
datadirs = [r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260202\f2\Trial4',
    r'Y:\Data\FCI\Hedwig\FB5AB_hDeltaC\260203\f1\Trial3']
meandat = np.zeros((600,2,len(datadirs)))
for di,d in enumerate(datadirs):
    cxa = CX_a(d,regions=regions2,yoking=False,denovo=False)
    fb5ab = cxa.pv2['0_fsbtn1_ch1'].to_numpy()
    #hdc = cxa.pv2['0_fsbtn1_ch2'].to_numpy()
    hdc = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
    jumps = cxa.get_entries_exits_like_jumps(odour='Oct')
    all_js = np.zeros((600,2,len(jumps)))
    for i,j in enumerate(jumps):
        dx = np.arange(j[0],j[2])
       # plt.plot(amp[dx],color='k',alpha=0.3)
        tamp = fb5ab[dx]
        amplen  = np.min([600,len(tamp)])
        all_js[:amplen,0,i] = tamp[:amplen]
        
        tamp = hdc[dx]
        amplen  = np.min([600,len(tamp)])
        all_js[:amplen,1,i] = tamp[:amplen]
        
    all_js[all_js==0] = np.nan    
    meandat[:,:,di] = np.nanmean(all_js,axis=2)
colours = ['r','b']
for i in range(2):
    plt.plot(x,meandat[:,i],color=colours[i])






