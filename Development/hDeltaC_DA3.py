# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:35:46 2025

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
#%% Image registraion

for i in [1,2,3,4]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\hDeltaC_DA3\250425\f1\Trial"+str(i))
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
r"Y:\Data\FCI\Hedwig\hDeltaC_DA3\250425\f1\Trial1",
r"Y:\Data\FCI\Hedwig\hDeltaC_DA3\250425\f1\Trial2",
r"Y:\Data\FCI\Hedwig\hDeltaC_DA3\250425\f1\Trial3",
r"Y:\Data\FCI\Hedwig\hDeltaC_DA3\250425\f1\Trial4"
                    ]
regions = ['fsb_upper']
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
    cxa = CX_a(e,regions=regions,denovo=True,yoking=False)
    
    
        
    
    cxa.save_phases()
    
#%%
experiment_dirs = [
r"Y:\Data\FCI\Hedwig\hDeltaC_DA3\250425\f1\Trial1",
r"Y:\Data\FCI\Hedwig\hDeltaC_DA3\250425\f1\Trial2",
r"Y:\Data\FCI\Hedwig\hDeltaC_DA3\250425\f1\Trial3",
r"Y:\Data\FCI\Hedwig\hDeltaC_DA3\250425\f1\Trial4"
                    ]
plt.close('all')
for e in experiment_dirs:
    cxa = CX_a(e,regions=['fsb_upper'],denovo=False,yoking=False)
    cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper'],yeseb=False)
    #cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
    
    plt.figure()
    wedges = cxa.pdat['wedges_fsb_upper']
    wamp = np.mean(wedges,axis=1)
    plt.plot(wamp,color='k')
    plt.plot(cxa.ft2['mfc3_stpt']/np.max(cxa.ft2['mfc3_stpt']),color='g')
    plt.plot(cxa.ft2['instrip'],color='r')
    
    try :
        plt.figure()
        cxa.mean_jump_arrows()
        cxa.mean_jump_lines()
    except:
        print('no jumps')