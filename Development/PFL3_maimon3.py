# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:51:57 2024

@author: dowel
"""

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
from Utils.utils_general import utils_general as ug
plt.rcParams['pdf.fonttype'] = 42 
#%% 
#%% Image registraion

for i in [2,3]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f2\Trial"+str(i))
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
    r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f1\Trial1",#Good behaviour but dim left LAL
    r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f1\Trial2",
    r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f2\Trial1",
    r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f2\Trial2",
    r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f2\Trial3"
                   ]

regions = ['LAL']
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
#%%
datadir = r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f2\Trial2"
cx = CX(name,regions,datadir)
pv2, ft, ft2, ix = cx.load_postprocessing()
#%%
L = pv2['0_lal']
R = pv2['1_lal']
ins = ft2['instrip']
plt.plot(L,color='b')
plt.plot(R,color='r')
plt.plot(ins,color='k')
plt.figure()
plt.plot(R-L,color='b')
plt.plot(ins,color='k')
plt.figure()
plt.plot(R+L,color='b')
plt.plot(ins,color='k')
#%% 
from analysis_funs.regression import fci_regmodel
fci = fci_regmodel(R-L,ft2,pv2)
#fci.rebaseline()
fci.example_trajectory_jump(cmin=-0.75,cmax=0.75)

fci = fci_regmodel((R+L)/2,ft2,pv2)
fci.rebaseline()
fci.example_trajectory_jump(cmin=-0.75,cmax=0.75)