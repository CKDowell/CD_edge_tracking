# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:45:56 2026

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
#r'Y:\Data\FCI\Hedwig\FB6H_SS950649_68A10_60D05_sytGCaMP8s_RCaMP3\260423\f2\Trial1',
        r'Y:\Data\FCI\Hedwig\FB6H_SS950649_68A10_60D05_sytGCaMP8s_RCaMP3\260423\f2\Trial2'    
            
            ]
regions = ['eb','fsb2','fsb1TN']
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
    regions2 = ['eb_ch1','fsb2_ch2','fsb1tn_ch1']
    cxa = CX_a(datadir,regions=regions2,yoking=True)
    cxa.save_phases()
#%%
fb6H = cxa.pv2['0_fsb1tn_ch1'].to_numpy()
hdc = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
ins = cxa.ft2['instrip'].to_numpy().astype(float)
plt.plot(fb6H,color='b')
plt.plot(hdc,color='k')
plt.plot(ins,color='r')

