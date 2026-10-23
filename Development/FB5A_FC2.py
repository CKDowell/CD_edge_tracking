# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:50:44 2026

@author: dowel
"""
from analysis_funs.regression import fci_regmodel
import os
import matplotlib.pyplot as plt 
from analysis_funs.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_tan import CX_tan
import pandas as pd
import numpy as np
from Utilities.utils_general import utils_general as ug
from analysis_funs.CX_analysis_col import CX_a
#%%




plt.close('all')
datadirs = [
 #r'Y:\Data\FCI\Hedwig\FC2_FB5A\261012\f1\Trial1',
   #  r'Y:\Data\FCI\Hedwig\FC2_FB5A\261012\f1\Trial2',
   # r'Y:\Data\FCI\Hedwig\FC2_FB5A\261012\f1\Trial3',
   #  r'Y:\Data\FCI\Hedwig\FC2_FB5A\261012\f1\Trial4',
    
    r'Y:\Data\FCI\Hedwig\FC2_FB5A\261015\f1\Trial1'
    
    ]
regions = ['fsb1','fsb2']
regions2 = ['fsb1_ch1','fsb2_ch2']
for e in datadirs:
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
    cxa = CX_a(datadir,regions=regions2,yoking=False)
    cxa.save_phases()
    plt.figure()
    cxa.simple_raw_plot(plotphase=False,regions = regions2,yeseb=False)
    
    
#%%
plt.close('all')
datadir = r'Y:\Data\FCI\Hedwig\FC2_FB5A\261015\f1\Trial1'
cxa = CX_a(datadir,regions=regions2,yoking=False,denovo=False)
cxa.simple_raw_plot(plotphase=False,regions = regions2,yeseb=False)
cxa.simple_raw_plot(plotphase=True,regions = regions2,yeseb=False)
fc2p = cxa.pdat['phase_fsb2_ch2']
fb5ap = cxa.pdat['phase_fsb1_ch1']
plt.figure()
plt.scatter(fc2p,fb5ap,color='k',s=1)


wedges = cxa.pdat['wedges_fsb1_ch1']

for i in range(wedges.shape[1]):
    wedges[:,i] = sg.savgol_filter(wedges[:,i],30,2)
fb5a2 = ug.phase_from_wed(wedges)

wedges = cxa.pdat['wedges_fsb2_ch2']
plt.figure()
plt.subplot(1,2,1)
plt.imshow(wedges,aspect='auto')
for i in range(wedges.shape[1]):
    wedges[:,i] = sg.savgol_filter(wedges[:,i],30,2)
fc22 = ug.phase_from_wed(wedges)
f = ug.phase_from_wed(cxa.pdat['wedges_fsb2_ch2'])
plt.subplot(1,2,2)
plt.imshow(wedges,aspect='auto')


plt.figure()
plt.scatter(fc22,fb5a2,color='k',s=1)

ins = cxa.ft2['instrip'].to_numpy()
plt.scatter(fc22[ins>0],fb5a2[ins>0],color='r',s=2)
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='b')
plt.plot([0,np.pi],[-np.pi,0],color='b')
plt.plot([-np.pi,0],[0,np.pi],color='b')
plt.figure()
plt.plot(fc22)
plt.plot(fb5a2)
plt.plot(ins,color='r')