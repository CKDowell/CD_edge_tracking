# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:34:06 2024

@author: dowel
"""

from analysis_funs.regression import fci_regmodel

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
from src.utilities import funcs as fn
plt.rcParams['pdf.fonttype'] = 42 
#%% Image registraion

for i in [3]:
    datadir =os.path.join("Y:\\Data\\FCI\\Hedwig\\FC2_maimon2\\240916\\f1\\Trial"+str(i))
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
    #"Y:\\Data\\FCI\\Hedwig\\FC2_maimon2\\240821\\f2\\Trial3"
    "Y:\\Data\\FCI\\Hedwig\\FC2_maimon2\\240916\\f1\\Trial3" # Switch from one to another but not enoughdata
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial5"# eb good swithc
                   ]
regions = ['fsb_lower','fsb_upper','pb']
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
    cx.save_postprocessing()
    pv2, ft, ft2, ix = cx.load_postprocessing()
    cxa = CX_a(datadir,regions=np.flipud(regions))
    
    
    cxa.save_phases()
#%%
datadir ="Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial5"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
#%%
plt.close('all')
cxa.simple_raw_plot(plotphase=True,regions = ['fsb_upper','fsb_lower'],yeseb = True,yk='eb')
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'],cxa.pdat['amp_fsb_upper']*2,a_sep=0.5)
cxa.plot_train_arrows(plumewidth=20,tperiod = 0.5)
plt.figure()
cxa.mean_phase_train(trng=1)
#%% Get summary of phase and heading for test and train epochs
train = ft2['intrain']
train[train.isnull()] = 0
train[train==False] = 0
plt.plot(train)

odour = ft2['mfc2_stpt']
instrip = ft2['instrip']
