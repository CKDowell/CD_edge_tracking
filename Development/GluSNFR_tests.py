# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 16:08:54 2025

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
from analysis_funs.CX_analysis_tan import CX_tan

plt.rcParams['pdf.fonttype'] = 42 

#%% Image registraion

for i in [1,2,3,4]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\FB5I_SS100553_iGluSNFR4\251010\f1","Trial" +str(i))
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
    r"Y:\Data\FCI\Hedwig\FB6A_SS95731_iGluSNFR\250911\f1\Trial1",
    r"Y:\Data\FCI\Hedwig\FB6A_SS95731_iGluSNFR\250911\f1\\Trial2",
   r"Y:\Data\FCI\Hedwig\FB5I_SS100553_iGluSNFR4\251008\f2\Trial5",               
   r"Y:\Data\FCI\Hedwig\FB5I_SS100553_iGluSNFR4\251008\f2\Trial3",   
r"Y:\Data\FCI\Hedwig\FB5I_SS100553_iGluSNFR4\251008\f1\Trial1",
r"Y:\Data\FCI\Hedwig\FB5I_SS100553_iGluSNFR4\251008\f1\Trial2",
r"Y:\Data\FCI\Hedwig\FB5I_SS100553_iGluSNFR4\251008\f1\Trial3",
r"Y:\Data\FCI\Hedwig\FB5I_SS100553_iGluSNFR4\251010\f1\Trial3"
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
    cx.process_rois(dynamicbaseline=True)
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()#upsample to 50Hz
    pv2, ft, ft2, ix = cx.load_postprocessing()
    cxa = CX_a(datadir,regions=regions,yoking=False)
    
    cxa.save_phases()
    
#%%
datadir =  r"Y:\Data\FCI\Hedwig\FB5I_SS100553_iGluSNFR4\251008\f2\Trial5"  
cxa = CX_a(datadir,regions=regions,yoking=False,denovo=False)
cxa.simple_raw_plot(plotphase=False,yeseb=False)
cxa.simple_raw_plot(plotphase=True,yeseb=False)
