# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:43:14 2026

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
    # r'Y:\Data\FCI\Hedwig\PFNa_EPG\260814\f1\Trial1', # Only one entry
    # r'Y:\Data\FCI\Hedwig\PFNa_EPG\260814\f1\Trial2', # inconsistent upwind vector # good tracking
    # r'Y:\Data\FCI\Hedwig\PFNa_EPG\260814\f1\Trial3', # Beautiful upwind vector of PFNa # more default mode like behaviour
    # r'Y:\Data\FCI\Hedwig\PFNa_EPG\260814\f1\Trial4' # Mostly upwind vector
    
    r'Y:\Data\FCI\Hedwig\PFNa_EPG\260818\f2\Trial1', # Mostly upwind but also downwind
    r'Y:\Data\FCI\Hedwig\PFNa_EPG\260818\f2\Trial2', # Mostly upwind but also downwind V good tracker
    r'Y:\Data\FCI\Hedwig\PFNa_EPG\260818\f2\Trial3' # Reverse heading signal
    
    
          ]

  

for datadir in datadirs:
   # regions = ['eb','fsb_upper_1','fsb_lower_1','fsb_upper_2']
    regions = ['eb','fsb']
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
    regions = regions = ['eb_ch2','fsb_ch1']
    #regions = ['eb_ch1','eb_ch2','fsb_upper_1_ch1','fsb_lower_1_ch1','fsb_upper_2_ch2']
    #regions = ['fsb_upper_ch1','fsb_upper_ch2','fsb_lower_ch1','fsb_lower_ch2']
    cxa = CX_a(datadir,regions=regions,yoking=True)
    cxa.save_phases()
    try:
        cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
    except:
        print('whoops')
        
#%%
plt.close('all')
datadir = r'Y:\Data\FCI\Hedwig\PFNa_EPG\260818\f2\Trial2'
regions = ['eb_ch2','fsb_ch1']
cxa = CX_a(datadir,regions=regions,yoking=True,denovo=False)
regions = ['eb_ch2','fsb_ch1']
cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=True)

colours = uplt.columnar_colours()
cxa.plot_traj_arrow_new(regions,a_sep=2,colours =colours,ascale=10)

plt.figure()
plt.hist(cxa.pdat['offset_fsb_ch1_phase'],bins=100)
plt.hist(cxa.pdat['offset_eb_ch2_phase'],bins=100)

plt.figure()
plt.scatter(cxa.pdat['offset_eb_ch2_phase'],cxa.pdat['offset_fsb_ch1_phase'],s=1)

plt.figure()
plt.scatter(cxa.ft2['ft_heading'],cxa.pdat['phase_eb_ch2'])
plt.pause(.1)
