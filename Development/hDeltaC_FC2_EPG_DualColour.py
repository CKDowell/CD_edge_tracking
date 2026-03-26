# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:20:37 2026

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
datadirs=[ 
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260323\f1\Trial1',#1030nm Not amazing behaviour plus shutter problem
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260323\f1\Trial2',#1030nm Not amazing behaviour
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260323\f1\Trial3', # 1020nm Not amazing behaviour
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260323\f1\Trial4' # 1020nm Did not make jumps but did a good number of entries
          ]

  

for datadir in datadirs:
   
    regions = ['eb','fsb1','fsb2','fsb1_me','fsb2_me']
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
    regions = regions = ['eb_ch1','fsb1_ch1','fsb1_ch2','fsb2_ch1','fsb2_ch2','fsb1_me_ch1','fsb1_me_ch2','fsb2_me_ch1','fsb2_me_ch2']
    #regions = ['eb_ch1','eb_ch2','fsb_upper_1_ch1','fsb_lower_1_ch1','fsb_upper_2_ch2']
    #regions = ['fsb_upper_ch1','fsb_upper_ch2','fsb_lower_ch1','fsb_lower_ch2']
    cxa = CX_a(datadir,regions=regions,yoking=True)
    cxa.save_phases()
    try:
        cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
    except:
        print('whoops')
#%%

regions = ['eb','fsb1','fsb2']

datadir =  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260323\f1\Trial3'
regions2 = ['eb_ch1','fsb1_ch1','fsb2_ch2']
cxa = CX_a(datadir,regions=regions2,yoking=True,denovo=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=True)
#%%
region1 = "fsb1_ch1"
region2 = "fsb2_ch2"
colours = colours = np.array([[49,99,125],[81,156,205]])/255
cxa.plot_traj_arrow_new([region2,region1],a_sep=5,colours =colours)


#%%
plt.figure()
fc2 = cxa.pdat['offset_fsb1_ch1_phase'].to_numpy()
hdc = cxa.pdat['offset_fsb2_ch2_phase'].to_numpy()
epg = cxa.pdat['offset_eb_ch1_phase'].to_numpy()
x = np.arange(0,len(fc2))/10
ins = cxa.ft2['instrip'].to_numpy()
plt.plot(x,ins,color='r')
plt.scatter(x,fc2,color=colours[0,:],s=3)
plt.scatter(x,hdc,color=colours[1,:],s=3)
plt.scatter(x,epg,color='k',s=3)


#%%
