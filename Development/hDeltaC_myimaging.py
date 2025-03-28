# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:37:50 2025

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

plt.rcParams['pdf.fonttype'] = 42 
#%% Image registraion

for i in [3,4]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250327\f2\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
#%%
experiment_dirs = [
    #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250313\f1\Trial1",
        #           r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250313\f1\Trial2",
                   
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f1\Trial1",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f1\Trial2",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f1\Trial3",
                   
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial1",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial2",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial3",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial4",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial5",
                  
                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250327\f2\Trial3",
                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250327\f2\Trial4"
                   ]
regions = ['eb','fsb_upper','fsb_lower']
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
    cxa = CX_a(datadir,regions=regions)
    
    
        
    
    cxa.save_phases()
    
#%%

experiment_dirs = [
    #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250313\f1\Trial1",#Lots of plume cross overs
    #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250313\f1\Trial2",#Lots of plume cross overs
                   
                   #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial1",#Not great behaviour
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial2",# Simple plume no jumps
                   #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial3",#Made a few jumps
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial4",#Octanol [?] pulses - neuron is inhibited
                   #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial5"#ACV pulses
                   
                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250327\f2\Trial3",
                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250327\f2\Trial4", #Octanol pulses files missing near end :( reanalyse
                   ]
plt.close('all')
for e in experiment_dirs:
    cxa = CX_a(e,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb')
    cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),cxa.pdat['amp_fsb_upper'],a_sep= 5)
    plt.figure()
    cxa.mean_jump_arrows()
    cxa.mean_jump_lines()







