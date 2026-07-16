# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:17:28 2026

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
    # r'Y:\Data\FCI\Hedwig\68A10_sytGC8s_PFNa_sytRC3\260615\f1\Trial1',
    # r'Y:\Data\FCI\Hedwig\68A10_sytGC8s_PFNa_sytRC3\260615\f1\Trial2',
    
    # r'Y:\Data\FCI\Hedwig\68A10_sytGC8s_PFNa_SS02255_sytRC3\260617\f2\Trial1',
    # r'Y:\Data\FCI\Hedwig\68A10_sytGC8s_PFNa_SS02255_sytRC3\260617\f2\Trial2',
    # r'Y:\Data\FCI\Hedwig\68A10_sytGC8s_PFNa_SS02255_sytRC3\260617\f2\Trial3',
    
    # r'Y:\Data\FCI\Hedwig\68A10_sytGC8s_PFNa_SS02255_sytRC3\260618\f2\Trial1',
    # r'Y:\Data\FCI\Hedwig\68A10_sytGC8s_PFNa_SS02255_sytRC3\260618\f2\Trial2',
    # r'Y:\Data\FCI\Hedwig\68A10_sytGC8s_PFNa_SS02255_sytRC3\260618\f2\Trial3',
    # r'Y:\Data\FCI\Hedwig\68A10_sytGC8s_PFNa_SS02255_sytRC3\260618\f2\Trial4',
    
    r'Y:\Data\FCI\Hedwig\PFNa_PFNpm\260714\f1\Trial1', #No hDeltaC but v good PFNa signal in ch1
    r'Y:\Data\FCI\Hedwig\PFNa_PFNpm\260714\f1\Trial2' # No hDeltaC but v good PFNa signal in ch1
    
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
plt.close('all')
datadir = r'Y:\Data\FCI\Hedwig\PFNa_PFNpm\260714\f1\Trial1'
cxa = CX_a(datadir,regions=regions,yoking=False,denovo=False)
regions = ['fsb1_ch1','fsb2_ch2']
cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=True)

pfna_p = cxa.pdat['phase_fsb1_ch1']
hdc_p = cxa.pdat['phase_fsb2_ch2'].squeeze()

plt.figure()
plt.scatter(pfna_p,hdc_p,color='k',s=1)

df = np.abs(ug.circ_subtract(hdc_p,pfna_p))
plt.figure()
plt.plot(df,color='k')
ins = cxa.ft2['instrip'].to_numpy()
plt.plot(ins,color='r')