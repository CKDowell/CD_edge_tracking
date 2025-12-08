# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 15:53:36 2025

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
    r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial1",
    r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial2",
    r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial3",
    r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial4",
    r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial5",
    
    r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f2\Trial2",
    r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f2\Trial3",
    r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f2\Trial4",
    r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f2\Trial5",
    
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
plt.close('all')
datadir =r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial1"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
cxa.simple_raw_plot(plotphase=False,regions = ['eb','fsb_upper','fsb_lower'],yk='eb')

cxa.simple_raw_plot(plotphase=True,regions = ['eb','fsb_upper','fsb_lower'],yk='eb')

cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 4)

#%%
  # if set(['train_heading']).issubset(self.ft2):
  #   plt.figure()
#cxa.mean_jump_arrows(fsb_names=['fsb_upper'],ascale=100,jsize=5)
#plt.ylim([-40,40])
idx = np.arange(8400,10100)
cxa.plot_traj_arrow_segment(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/5,axis=1),idx,a_sep= 2)