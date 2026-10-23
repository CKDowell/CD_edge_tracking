# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:56:58 2026

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

    
# r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\260922\f1\Trial1',
# r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\260922\f1\Trial2',
#   r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\260923\f1\Trial1', # Octanol pulses
#   r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\260923\f1\Trial2', # ACV pulses
  
  r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\261008\f1\Trial1',
  r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\261008\f1\Trial2',#ET
  r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\261008\f1\Trial3',# Octanol
  r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\261008\f1\Trial4', # wrong pulse interval
  r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\261008\f1\Trial5' #weird activity, maybe fly is dying.
                   ]

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