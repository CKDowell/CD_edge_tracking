# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:01:30 2024

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

import numpy as np
#%%
flies =['Y:\\Data\\FCI\\AndyData\\21D07\\20220103_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220107_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220110_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220111_21D07_sytjGCaMP7f_Fly1_001\\processed']
#%%
datadir = 'Y:\\Data\\FCI\\AndyData\\21D07\\20220103_21D07_sytjGCaMP7f_Fly1_001\\processed'
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
post_processing_file = os.path.join(datadir,'postprocessing.h5')
pv2 = pd.read_hdf(post_processing_file, 'pv2')
ft2 = pd.read_hdf(post_processing_file, 'ft2')
fc = fci_regmodel(pv2['fb5ab_dff'],ft2,pv2)
fc.example_trajectory(cmin=-0.2,cmax=0.2)