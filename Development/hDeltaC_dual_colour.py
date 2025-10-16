# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:08:43 2025

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

datadir =os.path.join(r"Y:\Data\FCI\Hedwig\60D05_68A10_RCaMP3_sytGCaMP8s\251001\f1\Test2")
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
regions = ['eb','fsb_upper','fsb_lower']
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
regions = ['eb_ch1','eb_ch2','fsb_upper_ch1','fsb_upper_ch2','fsb_lower_ch1','fsb_lower_ch2']
cxa = CX_a(datadir,regions=regions,yoking=True)

cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
