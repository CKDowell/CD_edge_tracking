# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:12:03 2024

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

for i in [1,4,5,6]:
    datadir =os.path.join("Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240806\\f1\\Trial"+str(i))
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
    "Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240805\\f1\\Trial3",
    "Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240806\\f1\\Trial1",
    "Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240806\\f1\\Trial4",
    "Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240806\\f1\\Trial6",
    "Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240806\\f1\\Trial5"
    
                   ]
for e in experiment_dirs:
    datadir =os.path.join(e)
    print(e)
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cx = CX(name,['fsb','eb'],datadir)
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois()
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()
    pv2, ft, ft2, ix = cx.load_postprocessing()

    try :
        cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'])
    except:
        cxa = CX_a(datadir,regions=['eb','fsb'])
    
    cxa.save_phases()
#%%
cxa = CX_a(experiment_dirs[-1],regions=['eb','fsb'],denovo=False)
#%%
plt.close('all')
cxa.simple_raw_plot(plotphase=True,regions = ['fsb'])
#cxa.simple_raw_plot(plotphase=True)
#%%
plt.close('all')
phase = cxa.pdat['offset_fsb_phase']
amp = np.mean(cxa.pdat['fit_wedges_fsb'],axis=1)/10

#cxa.plot_traj_arrow(phase,amp,a_sep=5)
for i in range(10):
    cxa.point2point_heat(i*1000,(i+1)*1000,regions=['eb','fsb'],toffset=0)