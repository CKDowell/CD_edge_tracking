# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:09:34 2025

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

for i in [1,4]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\FC2_PAM\250807\f1\Trial"+str(i))
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
  #r"Y:\Data\FCI\Hedwig\FC2_508E03\250219\f1\Trial1",
#             r"Y:\Data\FCI\Hedwig\FC2_508E03\250219\f1\Trial2",
           # r"Y:\Data\FCI\Hedwig\FC2_508E03\250219\f1\Trial3" # poor imaging signal.
           # r"Y:\Data\FCI\Hedwig\FC2_PAM\250804\f1\Trial1", # Oct plume, tracking
           # r"Y:\Data\FCI\Hedwig\FC2_PAM\250804\f1\Trial2"
           #r"Y:\Data\FCI\Hedwig\FC2_PAM\250805\f2\Trial1",
           #r"Y:\Data\FCI\Hedwig\FC2_PAM\250805\f2\Trial2",#Great edge tracking !!!
          # r"Y:\Data\FCI\Hedwig\FC2_PAM\250805\f2\Trial3",# Octanol, not great
         # r"Y:\Data\FCI\Hedwig\FC2_PAM\250806\f1\Trial2",# Great edge tracker!!! FC2 phAW largely in phase with EPG...
        #  r"Y:\Data\FCI\Hedwig\FC2_PAM\250806\f1\Trial3" # Octanol, not edge tracking. FC2 matches EPG perfectly. Weird...
          
          r"Y:\Data\FCI\Hedwig\FC2_PAM\250807\f1\Trial1",# ACV pulses with turns, alternating stims
          r"Y:\Data\FCI\Hedwig\FC2_PAM\250807\f1\Trial4" # Octanol pulses with turns, a bit of a mess behaviourally
            ]
regions = ['eb' ,'fsb_upper','fsb_lower']
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
    cx.save_postprocessing()
    pv2, ft, ft2, ix = cx.load_postprocessing()
    cxa = CX_a(datadir,regions=regions,yoking=True,stim=True)
    
    
    cxa.save_phases()
    
#%%  Plot data from ACV training
datadir = r"Y:\Data\FCI\Hedwig\FC2_PAM\250807\f1\Trial1"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],yoking=True,stim=True,denovo=False)
#%%
plt.close('all')
entries,exits = cxa.get_entries_exits()
heading= cxa.ft2['ft_heading']
phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
phase_eb = cxa.pdat['offset_eb_phase'].to_numpy()
led = cxa.ft2['led1_stpt'].to_numpy()
poff = ug.circ_subtract(phase,phase_eb)
for i,e in enumerate(entries):
    emn =stats.circmean(heading[e-10:e-5],np.pi,-np.pi)
    pmn = stats.circmean(phase[e-10:e-5],np.pi,-np.pi)
    ebmn = stats.circmean(phase_eb[e-10:e-5],np.pi,-np.pi)
    poffmn = stats.circmean(poff[e-10:e-5],np.pi,-np.pi)
    #plt.scatter(i,emn,color='k')
    plt.plot([i,i],[0,pmn],color='b')
    plt.scatter(i,ebmn,color=[0.2,0.2,0.2],s=10,zorder=10)
    
    plt.plot([i,i],[0,poffmn],color=[0.2,1,1])
    sign = np.sign(emn)
    ledval= led[e]
    if ledval==0:
        colour = [1,0.2,0.2]
    else:
        colour =[0.7,.7,.7]
    if 9==np.mod(i,10):
        
        plt.fill([i-9.5,i+0.5,i+0.5,i-9.5],[0,0,np.pi*sign,np.pi*sign],zorder=-1,color=colour)
        last9 = i
    if i==len(entries)-1:
        plt.fill([last9+.5,i+0.5,i+0.5,last9+.5],[0,0,np.pi*sign,np.pi*sign],zorder=-1,color=colour)
plt.ylim([-np.pi,np.pi])
plt.ylabel('Phase (rad)')
plt.xlabel('Plume entry number')


#%%
datadir = r"Y:\Data\FCI\Hedwig\FC2_PAM\250806\f1\Trial2"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],yoking=True,stim=True,denovo=False)
#cxa.ft2['instrip'] = cxa.ft2['mfc3_stpt']
#cxa.ft2['instrip'][cxa.ft2['instrip']<0.02] = 0
cxa.simple_raw_plot(regions=['fsb_upper','fsb_lower'],plotphase=True)
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
#%%  Phase nulled bumps
bins=5
plotdata = cxa.phase_nulled_jump(bins=bins,fsb_names=['eb','fsb_upper'],walk='led')
x = np.linspace(0,1,16)
for i in range(bins):
    plt.plot(x+i,plotdata[:,bins+i,1],color='k')
