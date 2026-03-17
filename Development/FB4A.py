# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 14:55:22 2026

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
#%% Image registraion

#%%
experiment_dirs = [
    #r'Y:\Data\FCI\Hedwig\FB4A\260210\f1\Trial1',
    #r'Y:\Data\FCI\Hedwig\FB4A\260210\f1\Trial2',
    r'Y:\Data\FCI\Hedwig\FB4A\260210\f1\Trial3',
   # r'Y:\Data\FCI\Hedwig\FB4A\260210\f1\Trial4'
                   
   
                   
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
    cxa = CX_a(datadir,regions=regions,yoking=False)
    
    cxa.save_phases()
    
    
    cxa.simple_raw_plot(regions=regions,yeseb=False)
    
#%%
plt.close('all')
for e in experiment_dirs:
    datadir =os.path.join(e)
    cxa = CX_a(datadir,regions=regions,yoking=False,denovo=False)

    cxa.simple_raw_plot(regions=regions,yeseb=False)
    cxa.simple_raw_plot(plotphase=True,regions=regions,yeseb=False)
    phase = cxa.pdat['phase_fsb']
    heading = cxa.ft2['ft_heading'].to_numpy()
    x = np.arange(0,len(heading))/10
    plt.figure()
    plt.scatter(x,phase,color='b',s=2)
    plt.scatter(x,heading,color='k',s=2)
    ps = ug.savgol_circ(phase,50,3)
    plt.plot(x,ps,color='b')
    plt.figure()
    plt.scatter(phase,heading,color='k',s=1)
    plt.xlim([-np.pi,np.pi])
    plt.ylim([-np.pi,np.pi])