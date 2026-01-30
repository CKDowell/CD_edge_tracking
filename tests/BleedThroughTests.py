# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:57:36 2026

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
#%% hDeltaC FSb test
datadir = r'Y:\Data\FCI\Hedwig\Tests\BleedThroughCheck2\Trial1'
regions = ['fsb']
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
regions = ['fsb_ch1','fsb_ch2']
cxa = CX_a(datadir,regions=regions,yoking=False)

cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
#%%
plt.close('all')
cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=True)

amp1 = np.mean(cxa.pdat['wedges_'+regions[0]],axis=1)
amp2 = np.mean(cxa.pdat['wedges_'+regions[1]],axis=1)
plt.figure()
plt.plot(amp1,color='r')
plt.plot(amp2,color='b')
plt.figure()
plt.scatter(cxa.pdat['phase_'+regions[0]],cxa.pdat['phase_'+regions[1]],s=1)

jumps = cxa.get_entries_exits_like_jumps()
colours = ['r','b']
plt.figure()
for a in range(2):
    
    all_js = np.zeros((600,len(jumps)))
    amp = np.mean(cxa.pdat['wedges_'+regions[a]],axis=1)
    for i,j in enumerate(jumps):
        dx = np.arange(j[1],j[2])
       # plt.plot(amp[dx],color='k',alpha=0.3)
        tamp = amp[dx]
        amplen  = np.min([600,len(tamp)])
        all_js[:amplen,i] = tamp[:amplen]
        
    all_js[all_js==0] = np.nan
    plt.plot(np.nanmean(all_js,axis=1),color=colours[a])
    
    
plt.figure()
corr = sg.correlate(amp1,amp2,mode='same')
corr = corr/np.max(corr)
plt.plot(corr)
amid = len(amp1)/2 
plt.plot([amid,amid],[0,1],color='r')

cmat = np.append(cxa.pdat['wedges_'+regions[0]],cxa.pdat['wedges_'+regions[1]],axis=1)


C = np.corrcoef(cmat.T)
plt.figure()
plt.imshow(C)
    
plt.figure()
p1 = cxa.pdat['phase_'+regions[0]]
p2 = cxa.pdat['phase_'+regions[1]]
t = np.arange(0,len(p1))/10
plt.scatter(t,p1,color='r',s=3)
plt.scatter(t,p2,color='b',s=3)    
    