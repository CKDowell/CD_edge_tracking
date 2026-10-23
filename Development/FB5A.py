# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:53:55 2026

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

#%% TN analysis
datadirs = [
    #r'Y:\Data\FCI\Hedwig\FB5A\260805\f1\Trial1',
   # r'Y:\Data\FCI\Hedwig\FB5A\260805\f1\Trial2',
    
    #r'Y:\Data\FCI\Hedwig\FB5A\260806\f1\Trial1',
   # r'Y:\Data\FCI\Hedwig\FB5A\260806\f1\Trial2',
    #r'Y:\Data\FCI\Hedwig\FB5A\260806\f1\Trial3', # Decent tracking
    # r'Y:\Data\FCI\Hedwig\FB5A\260806\f1\Trial4', # ACV pulses
    # r'Y:\Data\FCI\Hedwig\FB5A\260806\f1\Trial5', # Octanol pulses
   
    
   # r'Y:\Data\FCI\Hedwig\FB5A\260812\f1\Trial1',
   #  r'Y:\Data\FCI\Hedwig\FB5A\260812\f1\Trial2',
   r'Y:\Data\FCI\Hedwig\FB5A\260812\f1\Trial3', # v cool, active when flies are tracking, silent when flies are indifferent to plume
    r'Y:\Data\FCI\Hedwig\FB5A\260812\f1\Trial4',
   
    ]
for datadir in datadirs:
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cx = CX(name,['fsbTN'],datadir)
    # Process ROIs and saves csv
    cx.process_rois()
    
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()
    pv2, ft, ft2, ix = cx.load_postprocessing()

   
    cxt = CX_tan(datadir)
    cxt.reinit_fc()
    cxt.fc.example_trajectory(cmin=0,cmax=1)
   # cxt.fc.example_trajectory_jump(cxt.ca_no_nan.copy(),cxt.ft,cmin=0,cmax=1,cmap='magma')
    plt.figure()
    plt.plot(cxt.ca_no_nan)
    plt.plot(cxt.ft2['instrip'].to_numpy())
    plt.plot(cxt.ft2['mfc1_stpt'].to_numpy()==.2)
    
#%%
datadir =r'Y:\Data\FCI\Hedwig\FB5A\260806\f1\Trial3'
cxt = CX_tan(datadir)
cxa = CX_a(datadir,regions=['fsb'],yoking=False,denovo=False)
#%%
plt.close('all')
cxt.reinit_fc()
cxt.fc.example_trajectory(cmin=0,cmax=1)
cxt.fc.example_trajectory_jump(cxt.ca_no_nan.copy(),cxt.ft,cmin=0,cmax=1,cmap='inferno')

plt.figure()
t = cxt.pv2['relative_time'].to_numpy()
plt.plot(t,cxt.ca_no_nan,color='k')
#plt.plot(t,cxt.ft2['instrip'].to_numpy(),color='r')
ins = cxt.ft2['instrip'].to_numpy().astype(float)
plt.fill_between(t,ins,ins*0,color='r',linewidth=0)
u = ug()
_,_,dd = u.get_velocity( cxt.ft2['ft_posx'].to_numpy(), 
                         cxt.ft2['ft_posy'].to_numpy(), cxt.pv2['relative_time'].to_numpy())
plt.plot(t[1:],-1+dd/20,color=[.5,.5,.5])
plt.ylabel('dF/F0')
plt.xlabel('time (s)')
plt.xlim([0,t[-1]])

plt.figure()
plt.plot(t,cxt.ca_no_nan,color='k')
plt.plot(t,(cxt.ft2['mfc1_stpt'].to_numpy()==.2)&(ins>0),color='r')
plt.plot(t,(cxt.ft2['mfc1_stpt'].to_numpy()==.2)&(ins<1),color='g')
plt.ylabel('dF/F0')
plt.xlabel('time (s)')
plt.xlim([0,t[-1]])
plt.plot(t[1:],-1+dd/20,color=[.5,.5,.5])
cxa.simple_raw_plot(plotphase=True,regions = ['fsb'],yeseb=False)
#%% Columnar analysis
plt.close('all')
datadirs = [
   #  r'Y:\Data\FCI\Hedwig\FB5A\260805\f1\Trial1',
   # r'Y:\Data\FCI\Hedwig\FB5A\260805\f1\Trial2',
    
   #  r'Y:\Data\FCI\Hedwig\FB5A\260806\f1\Trial1',
   # r'Y:\Data\FCI\Hedwig\FB5A\260806\f1\Trial2',
   #  r'Y:\Data\FCI\Hedwig\FB5A\260806\f1\Trial3', # Decent tracking
   #  r'Y:\Data\FCI\Hedwig\FB5A\260806\f1\Trial4', # ACV pulses
   #  r'Y:\Data\FCI\Hedwig\FB5A\260806\f1\Trial5', # Octanol pulses
   
    
   # r'Y:\Data\FCI\Hedwig\FB5A\260812\f1\Trial1',
   #  r'Y:\Data\FCI\Hedwig\FB5A\260812\f1\Trial2',
   #  r'Y:\Data\FCI\Hedwig\FB5A\260812\f1\Trial3', # v cool, active when flies are tracking, silent when flies are indifferent to plume
   #  r'Y:\Data\FCI\Hedwig\FB5A\260812\f1\Trial4',
   
    r'Y:\Data\FCI\Hedwig\FB5A_60D05\260831\f1\Trial1',
    r'Y:\Data\FCI\Hedwig\FB5A_60D05\260831\f1\Trial2',
    r'Y:\Data\FCI\Hedwig\FB5A_60D05\260831\f1\Trial3',
    r'Y:\Data\FCI\Hedwig\FB5A_60D05\260831\f1\Trial4', # Strong walking, strange tracking, v good bump
    r'Y:\Data\FCI\Hedwig\FB5A_60D05\260831\f1\Trial5', # Odour pulse medley
    
    
   r'Y:\Data\FCI\Hedwig\FB5A_60D05\260831\f2\Trial1', # Not great walking or bump
   r'Y:\Data\FCI\Hedwig\FB5A_60D05\260831\f2\Trial2', # Not great walking or bump
   r'Y:\Data\FCI\Hedwig\FB5A_60D05\260831\f2\Trial3', # Not great walking or bump
   r'Y:\Data\FCI\Hedwig\FB5A_60D05\260831\f2\Trial4' # Not great walking or bump
    
   
    
    ]
regions = ['eb','fsb']
regions2 = ['eb_ch2','fsb_ch1']
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
    cxa = CX_a(datadir,regions=regions2,yoking=True)
    cxa.save_phases()
    plt.figure()
    cxa.simple_raw_plot(plotphase=False,regions = ['eb_ch2','fsb_ch1'],yeseb=False)
    
#%%
datadir = r'Y:\Data\FCI\Hedwig\FB5A_60D05\260831\f1\Trial4'
cxa = CX_a(datadir,regions=regions2,yoking=True,denovo=False)

cxa.simple_raw_plot(plotphase=True,regions = ['eb_ch2','fsb_ch1'],yeseb=False)
cxa.simple_raw_plot(plotphase=False,regions = ['eb_ch2','fsb_ch1'],yeseb=False)

pva = ug.get_pvas(cxa.pdat['wedges_fsb_ch1'])

cxa.plot_traj_arrow_heat([regions2[1]],pva,a_sep=5,colormap='coolwarm',cmin=0,cmax=.15)
#%%
plt.close('all')
for e in datadirs:
    cxa = CX_a(e,regions=regions2,denovo=False)
    cxa.simple_raw_plot(plotphase=False,regions = ['eb_ch2','fsb_ch1'],yeseb=False)
    plt.title(cxa.name)
    cxa.simple_raw_plot(plotphase=True,regions = ['eb_ch2','fsb_ch1'],yeseb=False)
    plt.title(cxa.name)
    # fmn = np.mean(cxa.pdat['wedges_fsb'],axis=1)
    # ins  = cxa.ft2['instrip'].to_numpy()
    # plt.figure()
    # plt.plot(fmn,color='k')
    # plt.plot(ins,color='r')
    # plt.title(cxa.name)
    # plt.figure()
    # fmnstandard = (fmn-np.mean(fmn))/np.std(fmn)
    # fc = fci_regmodel(fmnstandard,cxa.ft2,cxa.pv2)
    # fc.example_trajectory_scatter(fmn,cmin=0,cmax=1)
    # plt.title(cxa.name)
    # plt.figure()
    # plt.scatter(cxa.ft2['ft_heading'],cxa.pdat['phase_fsb'],s=1)
    # plt.title(cxa.name)
    # plt.xlabel('heading');plt.ylabel('fsb phase')
    # plt.figure()
    # t = cxa.pv2['relative_time'].to_numpy()
    # heading = cxa.ft2['ft_heading'].to_numpy()
    # phase = cxa.pdat['phase_fsb']
    # plt.scatter(t,heading,color='k',s=3)
    # plt.scatter(t,phase-np.pi*2,color='b',s=3)
    # plt.plot(t,ins*np.pi*3 -np.pi*2,color='r')
    # plt.title(cxa.name)
    
    
#%% old for single colour
from scipy import stats
plt.close('all')
for e in datadirs: 
    cxa = CX_a(e,regions=regions2,denovo=False)
    wedges = cxa.pdat['wedges_fsb']
    
    for i in range(wedges.shape[1]):
        wedges[:,i] = sg.savgol_filter(wedges[:,i],30,2)
    
    #wnull = ug.phase_nulling(wedges, cxa.pdat['phase_fsb'])
    ins  = cxa.ft2['instrip'].to_numpy()
    plt.figure()
    t = cxa.pv2['relative_time'].to_numpy()
    heading = cxa.ft2['ft_heading'].to_numpy()
    #phase = cxa.pdat['phase_fsb']
    phase = ug.phase_from_wed(wedges)
    offset = stats.circmean(ug.circ_subtract(phase,heading),low=-np.pi,high=np.pi)
    phase_off = ug.circ_subtract(phase,offset)
    
    plt.scatter(t,heading,color='k',s=3)
    plt.scatter(t,phase,color='b',s=3)
    plt.scatter(t,phase_off,color='m',s=3)
    plt.plot(t,ins*np.pi*3 -np.pi*2,color='r')
    plt.title(cxa.name)
    
    
    plt.figure()
    
    pva = ug.get_pvas(wedges)
    plt.plot(pva*10)
    plt.plot(ins)
    plt.title(cxa.name)
    
    plt.figure()
    plt.scatter(heading,phase,s=1)
    
    plt.scatter(heading,phase_off,s=1)
    plt.figure()
    plt.imshow(wedges,aspect='auto',interpolation='None')
    
#%%
plt.close('all')
for e in datadirs: 
    cxa = CX_a(e,regions=regions2,denovo=False)
    ins  = cxa.ft2['instrip'].to_numpy()
    plt.figure()
    t = cxa.pv2['relative_time'].to_numpy()
    
    wedges = cxa.pdat['wedges_fsb_ch1'].copy()
    
    for i in range(wedges.shape[1]):
        wedges[:,i] = sg.savgol_filter(wedges[:,i],30,2)
    phase = ug.phase_from_wed(wedges)
    fsb = ug.circ_subtract(phase,cxa.pdat['offset'].to_numpy())
    
    epg = cxa.pdat['offset_eb_ch2_phase'].to_numpy()
    fsb = cxa.pdat['offset_fsb_ch1_phase'].to_numpy()
    
    plt.scatter(t,epg,color='k',s=3)
    
    plt.scatter(t,fsb,color='b',s=3)
    plt.plot(t,ins,color='r')
    
    plt.figure()
    wmean = np.mean(wedges,axis=1)
    plt.plot(t,wmean,color='k')
    plt.plot(t,ins,color='r')
    
    cxa.simple_raw_plot(plotphase=False,regions = ['eb_ch2','fsb_ch1'],yeseb=False)
    cxa.simple_raw_plot(plotphase=True,regions = ['eb_ch2','fsb_ch1'],yeseb=False)
    
    
    
    










