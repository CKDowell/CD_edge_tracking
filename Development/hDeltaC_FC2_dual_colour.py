# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 17:36:52 2025

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

for i in [1,2]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251017\f1", "Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir,dual_color=True)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
    
    
#%%
datadirs=[ 
    #r"Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251014\f2\Trial1", # not great behaviour proof of principle
          #r"Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251017\f1\Trial1",# 20 entries plume traversals
         # r"Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251017\f1\Trial2",# some entries and exits
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251202\f1\Trial1',
         r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251202\f1\Trial2',
         r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251202\f1\Trial3'
         
          ]

  

for datadir in datadirs:
   # regions = ['eb','fsb_upper_1','fsb_lower_1','fsb_upper_2']
    regions = ['fsb1','fsb2']
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
    regions = regions = ['fsb1_ch1','fsb1_ch2','fsb2_ch1','fsb2_ch2']
    #regions = ['eb_ch1','eb_ch2','fsb_upper_1_ch1','fsb_lower_1_ch1','fsb_upper_2_ch2']
    #regions = ['fsb_upper_ch1','fsb_upper_ch2','fsb_lower_ch1','fsb_lower_ch2']
    cxa = CX_a(datadir,regions=regions,yoking=True)
    cxa.save_phases()
    try:
        cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
    except:
        print('whoops')
#%% 
datadir = r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251202\f1\Trial2'
regions2 = ['fsb1_ch1','fsb2_ch2']
cxa = CX_a(datadir,regions=regions,yoking=False,denovo=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=True)
#%%
colours = np.array([[81,61,204],[49,99,125],[81,156,205]])/255
phase_fc2 = cxa.pdat['phase_fsb1_ch1']
phase_hdc = cxa.pdat['phase_fsb2_ch2']
ins = cxa.ft2['instrip'].to_numpy()

x = np.arange(0,len(phase_fc2))
plt.scatter(x,phase_fc2,color=colours[2,:],s=3)
plt.scatter(x,phase_hdc,color=colours[1,:],s=3)
plt.plot(x,2*ins*np.pi-np.pi,color='r')
amp = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
amp = amp/np.std(amp)
plt.plot(x,amp-8,color=colours[1,:])
#%% In plume yoking
jumps = cxa.get_entries_exits_like_jumps(ent_duration=1.5)
phase_fc2 = cxa.pdat['phase_fsb1_ch1']
phase_hdc = cxa.pdat['phase_fsb2_ch2']
heading = cxa.ft2['ft_heading'].to_numpy()
offset = np.zeros_like(phase_fc2)
last = 0
for j in jumps:
    dx = np.arange(j[1]-10,j[1])
    theading = stats.circmean(heading[dx],low=-np.pi,high=np.pi)
    tphase = stats.circmean(phase_fc2[dx])
    off = ug.circ_subtract(tphase,theading)
    offset[last:j[1]] = off
    last = j[1]  
offset_fc2_phase = ug.circ_subtract(phase_fc2,offset)
offset_hdc_phase = ug.circ_subtract(phase_hdc,offset)
cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(offset_fc2_phase)
cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(offset_hdc_phase)
x = np.arange(0,len(phase_fc2))/10
plt.scatter(x,ug.savgol_circ(offset_fc2_phase,20,3),color=colours[2,:],s=3)
plt.scatter(x,ug.savgol_circ(offset_hdc_phase,20,3),color=colours[1,:],s=3)
plt.plot(x,2*ins*np.pi-np.pi,color='r')
amp = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
amp = amp/np.std(amp)
plt.plot(x,amp-8,color=colours[1,:])

#%%
regions2 = ['eb_ch1','eb_ch2','fsb_upper_1_ch1','fsb_upper_2_ch2']
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=True)
regions2 = ['eb_ch1','fsb_upper_1_ch1','fsb_upper_2_ch2']
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=False)

cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=True)

#%%
region1 = "fsb_upper_1_ch1"
cxa.plot_traj_arrow(cxa.pdat['offset_'+region1+'_phase'].to_numpy(),np.mean(cxa.pdat['wedges_'+region1]/2,axis=1),a_sep= 2)

region2 = "fsb_upper_2_ch2"
cxa.plot_traj_arrow(cxa.pdat['offset_'+region2+'_phase'].to_numpy(),np.mean(cxa.pdat['wedges_'+region2]/2,axis=1),a_sep= 2)


cxa.plot_traj_arrow_new([region2,region1],a_sep=5)
#%%
cxa.plot_traj_arrow_new(['fsb1_ch2','fsb1_ch1'],a_sep=10)
