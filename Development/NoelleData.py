# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:44:35 2024

@author: dowel
"""

#%%
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

#%% Imaging test for PFL3 neurons


datadir =os.path.join("F:\\noelle_imaging\\MBON30\\240423\\F1\\Trial1")
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
#%% Registration
ex = im.fly(name, datadir)
ex.register_all_images(overwrite=True)
ex.z_projection()
#%
ex = im.fly(name, datadir)
ex.mask_slice = {'All': [1,2,3,4]}
ex.t_projection_mask_slice()
#%% 

d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
cx = CX(name,['MBON30mask'],datadir)
# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()
pv2, ft, ft2, ix = cx.load_postprocessing()