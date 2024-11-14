# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:50:39 2024

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

for i in [1,2,3,4]:
    datadir =os.path.join("Y:\\Data\\FCI\\Hedwig\\FC2_maimon2iGluSNFR8880\\241112\\f2\\Trial"+str(i))
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
    "Y:\\Data\\FCI\\Hedwig\\FC2_maimon2iGluSNFR8880\\241112\\f2\\Trial1",
                   "Y:\\Data\\FCI\\Hedwig\\FC2_maimon2iGluSNFR8880\\241112\\f2\\Trial2",
                   "Y:\\Data\\FCI\\Hedwig\\FC2_maimon2iGluSNFR8880\\241112\\f2\\Trial3",
                   "Y:\\Data\\FCI\\Hedwig\\FC2_maimon2iGluSNFR8880\\241112\\f2\\Trial4"]
regions = ['fsb_whole','fsb']
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
    cx.save_postprocessing(uperiod=0.02)
    pv2, ft, ft2, ix = cx.load_postprocessing()
    cxa = CX_a(datadir,regions=['fsb'],yoking=False)
    cxa.save_phases()
#%%
cxa = CX_a(experiment_dirs[3],regions=['fsb'],yoking=False)

cxa.simple_raw_plot(plotphase=False,regions = ['fsb'],yeseb=False,yk='eb')
