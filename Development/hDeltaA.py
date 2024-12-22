# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:03:50 2024

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

for i in [1,2,3]:
    datadir =os.path.join("Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f1\\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
    
#%% To check
experiment_dirs = [
    # "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241120\\f1\\Trial1",
    # "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241120\\f2\\Trial1",
    # "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241120\\f2\\Trial2",
    
    "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f1\\Trial1",#2 jumps
    "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f1\\Trial2",#
    "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f1\\Trial3",
    "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f2\\Trial1",#5 jumps
    "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f2\\Trial2",
    
    
                   
                   ]
regions = ['fsb_lower','fsb_upper','eb']
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
    cxa = CX_a(datadir,regions=np.flipud(regions))
    plt.figure()
    cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb') 
    cxa.save_phases()
    
    
#%%
for e in experiment_dirs[3]:
    datadir =os.path.join(e)
    cxa = CX_a(datadir,regions=np.flipud(regions),denovo='False')
    cxa.simple_raw_plot(plotphase=True,regions = ['fsb_upper','fsb_lower'],yk='eb') 
    cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),cxa.pdat['amp_fsb_upper'],a_sep= 5)

#%%
datadir=  experiment_dirs[3]
cxa = CX_a(datadir,regions=np.flipud(regions),denovo='False')
cxa.simple_raw_plot(plotphase=True,regions = ['fsb_upper','fsb_lower'],yk='eb') 
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),cxa.pdat['amp_fsb_upper'],a_sep= 5)
cxa.mean_jump_lines()
    