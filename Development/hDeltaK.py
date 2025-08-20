# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 08:45:38 2025

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

plt.rcParams['pdf.fonttype'] = 42 
#%% Image registraion

for i in [1,2,3,4]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\hDeltaK_SS63089\250807\f2\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
 #%% Basic data processing
experiment_dirs = [r"Y:\Data\FCI\Hedwig\hDeltaK_SS63089\250807\f2\Trial1",
                   r"Y:\Data\FCI\Hedwig\hDeltaK_SS63089\250807\f2\Trial2",
                   r"Y:\Data\FCI\Hedwig\hDeltaK_SS63089\250807\f2\Trial3",
                   r"Y:\Data\FCI\Hedwig\hDeltaK_SS63089\250807\f2\Trial4"
    
                   ]

regions = ['eb','fsb_upper','fsb_lower']
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
    cxa = CX_a(datadir,regions=regions)
    cxa.save_phases()
#%%
cxa = CX_a(datadir,regions=regions,denovo=False)
cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb')
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),cxa.pdat['amp_fsb_upper'],a_sep= 5)
#%%
cxa.jump_return_details()

#%% 
x = np.arange(0,len(cxa.pdat['phase_fsb_upper']))/10
plt.scatter(x,cxa.pdat['phase_fsb_upper'],color='b',s=5)
#plt.scatter(x,cxa.pdat['phase_eb'],color='k',s=2)
eb180 = ug.circ_subtract(cxa.pdat['phase_eb'],np.pi)
eb180_filt = ug.savgol_circ(eb180,40,3)
plt.scatter(x,eb180,color=[0,0,0],s=5)
plt.scatter(x,eb180_filt,color=[0.2,0.2,0.2],s=5)   
    

#%%  Time lag
eb180_diff = ug.circ_vel(eb180,x,smooth=False,winlength=10)
fsb_diff = ug.circ_vel(cxa.pdat['phase_fsb_upper'],x,smooth=False,winlength=10)
c = sg.correlate(eb180_diff,fsb_diff)
c= c/np.max(c)
lags = sg.correlation_lags(len(eb180_diff),len(fsb_diff))/10
plt.plot(lags,c)
plt.plot([0,0],[0,1],color='k',linestyle='--')
plt.xlim([-5,5])
plt.xticks(np.arange(-5,5))
plt.xlabel('Time lag')
pk = np.argmax(c[:int(len(c)/2-2.5)])
plt.scatter(lags[pk],1)
plt.text(lags[pk]-1,1,str(lags[pk]))
plt.ylabel('Cross correlation')
    