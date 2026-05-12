# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:24:39 2026

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
datadirs=[ 
    # r'Y:\Data\FCI\Hedwig\PFNpm_83D12_FC2_sytGCaMP8s_RCaMP\260423\f1\Trial1', # image quality not amazing
    # r'Y:\Data\FCI\Hedwig\PFNpm_83D12_FC2_sytGCaMP8s_RCaMP\260423\f1\Trial2', # image quality not amazing
    # r'Y:\Data\FCI\Hedwig\PFNpm_83D12_FC2_sytGCaMP8s_RCaMP\260423\f1\Trial3' # image quality not amazing
    
    # r'Y:\Data\FCI\Hedwig\PFNpm_83D12_FC2_sytGCaMP8s_RCaMP\260425\f1\Trial1', # may not be FC2
    # r'Y:\Data\FCI\Hedwig\PFNpm_83D12_FC2_sytGCaMP8s_RCaMP\260425\f1\Trial2', #  may not be FC2
    # r'Y:\Data\FCI\Hedwig\PFNpm_83D12_FC2_sytGCaMP8s_RCaMP\260425\f1\Trial3' #  may not be FC2
    
    r'Y:\Data\FCI\Hedwig\PFNpm_83D12_FC2_sytGCaMP8s_RCaMP\260426\f2\Trial1',
    r'Y:\Data\FCI\Hedwig\PFNpm_83D12_FC2_sytGCaMP8s_RCaMP\260426\f2\Trial2',
    r'Y:\Data\FCI\Hedwig\PFNpm_83D12_FC2_sytGCaMP8s_RCaMP\260426\f2\Trial3' 
          ]

  

for datadir in datadirs:
   # regions = ['eb','fsb_upper_1','fsb_lower_1','fsb_upper_2']
    regions = ['fsb1','fsb2']
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
    regions = regions = ['fsb1_ch1','fsb1_ch2','fsb2_ch1','fsb2_ch2']
    #regions = ['eb_ch1','eb_ch2','fsb_upper_1_ch1','fsb_lower_1_ch1','fsb_upper_2_ch2']
    #regions = ['fsb_upper_ch1','fsb_upper_ch2','fsb_lower_ch1','fsb_lower_ch2']
    cxa = CX_a(datadir,regions=regions,yoking=False)
    cxa.save_phases()
    try:
        cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
    except:
        print('whoops')

#%%        
regions = ['fsb1','fsb2']

datadir =   r'Y:\Data\FCI\Hedwig\PFNpm_83D12_FC2_sytGCaMP8s_RCaMP\260426\f2\Trial2'
regions2 = ['fsb1_ch1','fsb2_ch2']
cxa = CX_a(datadir,regions=regions2,yoking=True,denovo=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=False)
#cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=True)

#%%
plt.close('all')
fc2 = cxa.pdat['phase_fsb1_ch1']
pfn = cxa.pdat['phase_fsb2_ch2']
heading = cxa.ft2['ft_heading'].to_numpy()
x = np.arange(0,len(fc2))/10
ins = cxa.ft2['instrip'].to_numpy()
plt.scatter(x,fc2,color='b',s=2)
plt.scatter(x,pfn,color='g',s=2)
plt.plot(x,ins,color='r')
plt.plot(x,heading,color='k')

#%% 
ins = cxa.ft2['instrip'].to_numpy()

phase = cxa.pdat['phase_fsb2_ch2'].squeeze()
phase1 = cxa.pdat['phase_fsb1_ch1'].squeeze()
heading = cxa.ft2['ft_heading'].to_numpy()
plt.figure()
plt.scatter(heading,phase,color='k',s=1)
plt.figure()
plt.scatter(heading,phase1,color='k',s=1)
plt.scatter(heading[ins>0],phase[ins>0],color='r',s=1)
plt.figure()
plt.scatter(phase,phase1,color='k',s=1)
plt.figure()
plt.scatter(phase[ins>0],phase1[ins>0],color='k',s=1)

u = ug()
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),np.arange(0,len(phase))/10)
vd = np.append(0,vd)
plt.figure()
plt.scatter(heading[vd>3],phase[vd>3],color='k',s=1)

