# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 10:16:46 2026

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
    
    
    #Relatively poor FC2 signal from this fly
    # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260918\f1\Trial1',
    # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260918\f1\Trial2',
 #    r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260918\f1\Trial3',
 #    r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260918\f1\Trial4',
 # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260918\f1\Trial5'
 
 # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f1\Trial1',
 # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f1\Trial2', # 12 jumps
 # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f1\Trial3'
 
 # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f2\Trial1',
 # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f2\Trial2',
 # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f2\Trial3',
 
 
 #r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260923\f1\Trial1',
# r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260923\f1\Trial2', # 7 jumps, FC2 and hDeltaJ highly correlated
# r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260923\f1\Trial3'
 
r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\261008\f1\Trial1', #No ET
r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\261008\f1\Trial2', # No ET but good walking hDeltaJ has strong backwards vector
r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\261008\f1\Trial3', #No Et charging through plumes. FC2 all over the place
 r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\261008\f1\Trial4' # No Et
  
    ]

  

for datadir in datadirs:
   
    regions = ['fsb1_lower','fsb2']#,'fsb1_upper']
    #regions = ['fsb1_lower','fsb2']
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
    #regions = regions = ['fsb1_lower_ch1','fsb1_upper_ch1','fsb2_ch2']
    regions = regions = ['fsb1_lower_ch1','fsb2_ch2']
    #regions = ['eb_ch1','eb_ch2','fsb_upper_1_ch1','fsb_lower_1_ch1','fsb_upper_2_ch2']
    #regions = ['fsb_upper_ch1','fsb_upper_ch2','fsb_lower_ch1','fsb_lower_ch2']
    cxa = CX_a(datadir,regions=regions,yoking=False)
    cxa.save_phases()
    try:
        cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
    except:
        print('whoops')
        
#%%

datadirs=[ 
    
    
    #Relatively poor FC2 signal from this fly
 #    r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260918\f1\Trial1',
 #    # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260918\f1\Trial2',
 #    r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260918\f1\Trial3',
 #    r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260918\f1\Trial4',
 # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260918\f1\Trial5'
 #r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f1\Trial1',
 r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f1\Trial2',
 r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f1\Trial3'
  
    ]

plt.close('all')

colours = uplt.columnar_colours()
regions = ['fsb1_lower_ch1','fsb2_ch2']
for datadir in datadirs:
    cxa = CX_a(datadir,regions=regions,yoking=False,denovo=False)
    cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
    cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=True)
    plt.figure()
    plt.scatter(cxa.pdat['phase_fsb1_lower_ch1'],cxa.ft2['ft_heading'],s=1)
    plt.figure()
    plt.scatter(cxa.pdat['phase_fsb1_lower_ch1'],cxa.pdat['phase_fsb2_ch2'],s=1)
    plt.figure()
    plt.scatter(cxa.pdat['phase_fsb2_ch2'],cxa.ft2['ft_heading'],s=1)
    hdj = cxa.pdat['phase_fsb1_lower_ch1']
    fc2 = cxa.pdat['phase_fsb2_ch2']
    t = np.arange(0,len(hdj))/10
    ins = cxa.ft2['instrip'].to_numpy()
    plt.figure()
    plt.scatter(t,hdj,color=colours[0,:],s=5)
    plt.scatter(t,fc2,color=colours[2,:],s=5)
    plt.plot(t,ins,color='r')
    
#%% In depth assessment
datadir= r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\261008\f1\Trial4'
#datadir = r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260923\f1\Trial2'
regions = ['fsb1_lower_ch1','fsb2_ch2']
cxa = CX_a(datadir,regions=regions,yoking=False,denovo=False)

#%%
# Centre FC2 based on in odour phase

all_jumps = cxa.get_entries_exits_like_jumps(ent_duration=1)
fc2 = cxa.pdat['phase_fsb2_ch2'].squeeze()

hdj = cxa.pdat['phase_fsb1_lower_ch1']
heading = cxa.ft2['ft_heading'].to_numpy()
aphase = np.zeros((len(all_jumps),3))
duration = 5
running_offset = np.zeros_like(heading)
last = 0
for ij,j in enumerate(all_jumps):
    dx = np.arange(j[0],j[1])
    p = stats.circmean(fc2[dx[-duration:]],high=np.pi,low=-np.pi)
    h = stats.circmean(heading[dx[-duration:]],high=np.pi,low=-np.pi)
    off = ug.circ_subtract(p,h)
    running_offset[last:j[2]] = off
    aphase[ij,:]= np.array([off,p,h])
    last = j[2]
running_offset[last:] = off
t = np.arange(0,len(aphase))
plt.scatter(aphase[:,1],aphase[:,2],s=10,c=t,cmap='coolwarm')
plt.plot([-np.pi,np.pi],[-np.pi,np.pi])
omean =stats.circmean(off,high=np.pi,low=-np.pi)
cxa.pdat['offset_fsb1_lower_ch1_phase'] = pd.Series( ug.circ_subtract(hdj,omean))
cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series( ug.circ_subtract(fc2,omean))
cxa.amp_eb = np.zeros_like(hdj)
colours = uplt.columnar_colours()
cxa.plot_traj_arrow_new(regions,a_sep=5,colours=colours[[0,2],:])

plt.figure()
plt.plot(running_offset)

plt.figure()
fc2 = cxa.pdat['offset_fsb2_ch2_phase']
hdj = cxa.pdat['offset_fsb1_lower_ch1_phase']
ins = cxa.ft2['instrip'].to_numpy()
t = np.arange(0,len(ins))/10
# plt.plot(t,ins,color='r')
plt.scatter(t,1+fc2/np.pi,color=colours[2,:],s=3)
plt.scatter(t,1+hdj/np.pi,color=colours[0,:],s=3)
u =  ug()
dx,dy,dd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),t)
da = 1+ug.circ_subtract(fc2,hdj)/np.pi
#plt.plot(t,da,color='k')
#plt.scatter(t,da,color='k',s=5)
plt.plot(t,ins*2,color='r')
plt.plot(t[1:],-2+dd/10,color=[.5,.5,.5])
jumps = cxa.get_entries_exits_like_jumps()
amp = np.mean(cxa.pdat['wedges_fsb1_lower_ch1'],axis=1)
pva = ug.get_pvas(cxa.pdat['wedges_fsb1_lower_ch1'])
plt.plot(t,amp*2-1,color=colours[0,:])
for ij,j in enumerate(jumps):
    plt.text(t[j[0]],np.pi+.3,ij)
#%%
# Centre based upon hDeltaJ default mode
all_jumps = cxa.get_entries_exits_like_jumps(ent_duration=.5)
jthresh = 100
jsize = all_jumps[:,2]-all_jumps[:,1]

tj = jsize>jthresh
all_jumps = all_jumps[tj,:]
aphase = np.zeros((len(all_jumps),3))
duration = 20
for ij,j in enumerate(all_jumps):
    dx = np.arange(j[1],j[2])
    p = stats.circmean(hdj[dx[-duration:]],high=np.pi,low=-np.pi)
    h = stats.circmean(heading[dx[-duration:]],high=np.pi,low=-np.pi)
    off = stats.circmean(ug.circ_subtract(hdj[dx[-duration:]],heading[dx[-duration:]]),high=np.pi,low=-np.pi)
    aphase[ij,:]= np.array([off,p,h])
    
t = np.arange(0,len(aphase))
plt.scatter(aphase[:,1],aphase[:,2],s=10,c=t,cmap='coolwarm')
plt.plot([-np.pi,np.pi],[-np.pi,np.pi])

#%% jumper comparison

"""
Jumper notes
1. 12 jumps, FC2 has a plume offset heading signal, matches hDJ early on as expected and not later on,
hDJ decays to a reverse heading signal pretty quickly

2. FC2 signal not a clear goal. High correlation between FC2 and hdJ at points. A bit of a worry.
    During jumps the two are not that well correlated, which is interesting. 

"""


datadirs = [ 
    r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f1\Trial2', # 12 jumps
 r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260923\f1\Trial2' # 7 jumps
 ]
for datadir  in datadirs:


    from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper
    etp = ET_paper(datadir,regions=regions)
    cxa = etp.cxa
    all_jumps = cxa.get_entries_exits_like_jumps(ent_duration=.75)
    fc2 = cxa.pdat['phase_fsb2_ch2'].squeeze()

    hdj = cxa.pdat['phase_fsb1_lower_ch1']
    heading = cxa.ft2['ft_heading'].to_numpy()
    aphase = np.zeros((len(all_jumps),3))
    duration = 5
    for ij,j in enumerate(all_jumps):
        dx = np.arange(j[0],j[1])
        p = stats.circmean(fc2[dx[-duration:]],high=np.pi,low=-np.pi)
        h = stats.circmean(heading[dx[-duration:]],high=np.pi,low=-np.pi)
        off = ug.circ_subtract(p,h)
        aphase[ij,:]= np.array([off,p,h])

    omean =stats.circmean(off,high=np.pi,low=-np.pi)
    cxa.pdat['offset_fsb1_lower_ch1_phase'] = pd.Series( ug.circ_subtract(hdj,omean))
    cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series( ug.circ_subtract(fc2,omean))
    cxa.pdat['offset_eb_phase'] = pd.Series( ug.circ_subtract(fc2,omean)*0.0)
    cxa.pdat['amp_eb'] =ug.circ_subtract(fc2,omean)*0.0
    etp.cxa = cxa
    plt.figure()
    etp.plt_tmp(regions =regions,colours=colours[[0,2],:],phase_num=10,scalar=5)
    cxa.mean_jump_lines(fsb_names=regions)

#%% Circular difference
d = ug.circ_subtract(cxa.pdat['phase_fsb2_ch2'].squeeze(),cxa.pdat['phase_fsb1_lower_ch1'].squeeze())
d = np.abs(d)
plt.plot(d)
r = ug.rowwise_pearson_smth(cxa.pdat['wedges_fsb2_ch2'], cxa.pdat['wedges_fsb1_lower_ch1'],5)
plt.plot(r)
plt.plot(ins,color='r')
cxa.plot_traj_arrow_heat(regions, r,cmin=-.6,cmax=.6)









