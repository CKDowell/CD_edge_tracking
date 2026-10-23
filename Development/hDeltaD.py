# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:39:28 2026

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


plt.rcParams['pdf.fonttype'] = 42

#%% 
# Summary no clear bump or plume modulation
experiment_dirs = [
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260819\f1\Trial1',
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260819\f1\Trial2',
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260819\f1\Trial3', # ACV pulses 
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260819\f1\Trial4', # Octanol pulses
  
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260819\f2\Trial1',
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260819\f2\Trial2',
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260819\f2\Trial3',# Octanol pulses
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260819\f2\Trial4',# ACV pulses 
  
  
  
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260820\f1\Trial1',
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260820\f1\Trial2',
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260820\f1\Trial3', #ACV pulses
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260820\f1\Trial4', # Oct pulses
  
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260820\f2\Trial1',
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260820\f2\Trial2',
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260820\f2\Trial3',# Oct pulses
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260820\f2\Trial4', #ACV pulses
  
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260825\f1\Trial1',
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260825\f1\Trial2',
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260825\f1\Trial3', #ACV pulses
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260825\f1\Trial4', #Oct pulses
  
  r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260826\f1\Trial1',
  r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260826\f1\Trial2',
  r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260826\f1\Trial3', # Oct pulses
  r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260826\f1\Trial4', # ACV pulses
  
  
  
                   ]

regions = ['fsb','eb']
regions = ['fsb']
for e in experiment_dirs:
    datadir =os.path.join(e)
    print(e)
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
    cxa = CX_a(datadir,regions=np.flipud(regions),yoking=False)
    cxa.save_phases()
    plt.figure()
    cxa.simple_raw_plot(plotphase=False,regions = ['fsb'],yeseb=False)
    
#%%
plt.close('all')
for e in experiment_dirs:
    cxa = CX_a(e,regions=np.flipud(regions),denovo=False)
    cxa.simple_raw_plot(plotphase=False,regions = ['fsb'],yeseb=False)
    plt.title(cxa.name)
    cxa.simple_raw_plot(plotphase=True,regions = ['fsb'],yeseb=False)
    plt.title(cxa.name)
    fmn = np.mean(cxa.pdat['wedges_fsb'],axis=1)
    ins  = cxa.ft2['instrip'].to_numpy()
    plt.figure()
    plt.plot(fmn,color='k')
    plt.plot(ins,color='r')
    plt.title(cxa.name)
    plt.figure()
    fmnstandard = (fmn-np.mean(fmn))/np.std(fmn)
    fc = fci_regmodel(fmnstandard,cxa.ft2,cxa.pv2)
    fc.example_trajectory_scatter(fmn,cmin=0,cmax=1)
    plt.title(cxa.name)
    #cxa.plot_traj_arrow(cxa.pdat['offset_fsb_phase'].to_numpy(),cxa.pdat['amp_fsb'],a_sep= 2)
    # plt.figure()
    # cxa.mean_jump_arrows(fsb_names=['fsb_upper'])
    # cxa.mean_jump_lines()