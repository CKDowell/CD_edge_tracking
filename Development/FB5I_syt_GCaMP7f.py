# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:13:29 2024

@author: dowel
"""

from analysis_funs.regression import fci_regmodel
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im

from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_tan import CX_tan
import numpy as np
from analysis_funs.CX_analysis_col import CX_a

#%% Imaging 


for i in [2,3]:
    datadir =os.path.join("Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553_sytGC7f\\240607\\f1\\Trial"+str(i))
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
datadir = "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553_sytGC7f\\240607\\f1\\Trial3"
cx = CX(name,['fsbTN'],datadir)
# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()
pv2, ft, ft2, ix = cx.load_postprocessing()

#%%
cxt = CX_tan(datadir,span=100)
cxt.fc.example_trajectory(cmin=-0.5,cmax =0.5)
#%% Columnar


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