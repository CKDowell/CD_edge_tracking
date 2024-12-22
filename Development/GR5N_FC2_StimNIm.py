# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:09:15 2024

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
from Utils.utils_general import utils_general as ug
plt.rcParams['pdf.fonttype'] = 42 
#%% Image registraion

for i in [3,4]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\CsChrimson_GR5a_FC2_maimon2\241217\f1\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
    
#%% Process ROIs

regions = ['fsb_upper']
experiment_dirs = [
    r"Y:\Data\FCI\Hedwig\CsChrimson_GR5a_FC2_maimon2\241217\f1\Trial3",
    r"Y:\Data\FCI\Hedwig\CsChrimson_GR5a_FC2_maimon2\241217\f1\Trial4"]
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
    cxa = CX_a(datadir,regions=np.flipud(regions),yoking=False)
#%%
datadir = r"Y:\Data\FCI\Hedwig\CsChrimson_GR5a_FC2_maimon2\241217\f1\Trial4"
cxa = CX_a(datadir,regions=['fsb_upper'],yoking=False)
cxa.simple_raw_plot(yeseb=False,regions=['fsb_upper'])