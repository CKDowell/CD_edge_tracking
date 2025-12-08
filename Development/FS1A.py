# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 18:33:34 2025

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
from analysis_funs.CX_analysis_tan import CX_tan
from Utilities.utils_general import utils_general as ug
plt.rcParams['pdf.fonttype'] = 42 

#%% Image registraion

#%%
experiment_dirs = [
    r'Y:\Data\FCI\Hedwig\FS1A_IS71375\251107\f1\Trial2',
r'Y:\Data\FCI\Hedwig\FS1A_IS71375\251107\f1\Trial3'
                   ]
regions = ['fsb','axon']
for e in experiment_dirs:
    datadir =os.path.join(e)
    print(e)
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    
    cx = CX(name,regions,datadir)
    
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois(dynamicbaseline=True)
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()#upsample to 50Hz
    pv2, ft, ft2, ix = cx.load_postprocessing()
    cxa = CX_a(datadir,regions=regions,yoking=False)
    
    cxa.save_phases()
#%%
ca = cxa.pv2['0_axon']
ca1= cxa.pv2['1_axon']
#ca2 = np.mean(cxa.pdat['wedges_fsb'],axis=1)
ins = cxa.ft2['instrip'].to_numpy()
u = ug()
_,_,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'].to_numpy())
vdn = vd/np.std(vd)

plt.plot(ca,color='k')
plt.plot(ca1,color=[0,0,0.3])
plt.plot(ca2*3,color=[0,0,0.6])
plt.plot(ins,color='r')
plt.plot(vdn/2-2,color=[0.3,.3,.3])
#%% 
phase = cxa.pdat['phase_fsb']
x = np.arange(0,len(phase))
plt.figure()
plt.scatter(x,phase,s=3,color='b')
heading  = cxa.ft2['ft_heading'].to_numpy()
plt.scatter(x,heading,s=3,color='k')
plt.plot(x,ins-3,color='r')
