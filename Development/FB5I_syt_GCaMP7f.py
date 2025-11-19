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
#%% Columnar processing
datadir = r"Y:\Data\FCI\Hedwig\FB5I_SS100553_sytGC7f\251104\f1\Trial2"
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
cx = CX(name,['fsb'],datadir)
# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()
pv2, ft, ft2, ix = cx.load_postprocessing()

#%% Simple TN processing
cx = CX(name,['fsbTN'],datadir)
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
#%%
pv2 = cxa.pv2
y = np.mean(pv2.filter(regex='fsb'),axis=1)
fc = fci_regmodel(y.to_numpy().flatten(),ft2,pv2)
fc.rebaseline()
fc.example_trajectory_jump(cmin=-0.2,cmax=0.2)
#%%
datadir = r"Y:\Data\FCI\Hedwig\FB5I_SS100553_sytGC7f\251104\f1\Trial2"


#cx = CX(name,['eb','fsb'],datadir) 
cxa = CX_a(datadir,regions=['fsb'],yoking=False)
cxa.simple_raw_plot(plotphase=False,regions = ['fsb'],yeseb=False)

cxa.simple_raw_plot(plotphase=True,regions = ['fsb'],yeseb=False)

phase = cxa.pdat['phase_fsb']
ins = cxa.ft2['instrip'].to_numpy()
x = np.arange(0,len(phase))
plt.scatter(x,phase,s=2,color='r')
plt.plot(x,ins,color='b')
