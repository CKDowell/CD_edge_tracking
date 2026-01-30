# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 17:01:59 2026

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
experiment_dirs = [
    #r'Y:\Data\FCI\Hedwig\Tests\60D05_GtACR2_Check\260127\f1\Trial1',
                   # r'Y:\Data\FCI\Hedwig\Tests\60D05_GtACR2_Check\260127\f1\Trial3',
                   # r'Y:\Data\FCI\Hedwig\Tests\60D05_GtACR2_Check\260127\f1\Trial4'
                   r'Y:\Data\FCI\Hedwig\Tests\60D05_GtACR2_Check\260127\f1\Trial5',
                   r'Y:\Data\FCI\Hedwig\Tests\60D05_GtACR2_Check\260127\f1\Trial6',
                   r'Y:\Data\FCI\Hedwig\Tests\60D05_GtACR2_Check\260127\f1\Trial7'
     ]

regions = ['eb']
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
    cxa = CX_a(datadir,regions=regions,yoking=False)
    
    cxa.save_phases()
    
    cxa.simple_raw_plot(regions=['eb'],yeseb=False)
