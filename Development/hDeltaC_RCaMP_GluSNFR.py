# -*- coding: utf-8 -*-
"""
Created on Wed May 14 13:49:22 2025

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
#%% Image registraion

for i in [1,2]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\hDeltaC_RCaMP_iGluSNFR8880\250715\f2\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir,dual_color=True)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
    
#%%
datadir= r"Y:\Data\FCI\Hedwig\hDeltaC_RCaMP_iGluSNFR8880\250513\f1\Trial2"

  
regions = ['fsb_upper','fsb_lower']
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
regions = ['fsb_upper_ch1','fsb_upper_ch2','fsb_lower_ch1','fsb_lower_ch2']
cxa = CX_a(datadir,regions=regions,yoking=False)

cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
#%% Quick load
datadir= r"Y:\Data\FCI\Hedwig\hDeltaC_RCaMP_iGluSNFR8880\250715\f2\Trial2"

regions = ['fsb_upper_ch1','fsb_upper_ch2','fsb_lower_ch1','fsb_lower_ch2']
cxa = CX_a(datadir,regions=regions,yoking=False)

#%%
w1 = np.mean(cxa.pdat['wedges_fsb_upper_ch1'],axis=1)
w2 = np.mean(cxa.pdat['wedges_fsb_upper_ch2'],axis=1)
ins = cxa.ft2['instrip']
plt.plot(w1,color='r')
plt.plot(w2,color='g')
plt.plot(ins,color='k')

plt.figure()
w1 = np.mean(cxa.pdat['wedges_fsb_lower_ch1'],axis=1)
w2 = np.mean(cxa.pdat['wedges_fsb_lower_ch2'],axis=1)
ins = cxa.ft2['instrip']
plt.plot(w1,color='r')
plt.plot(w2,color='g')
plt.plot(ins,color='k')