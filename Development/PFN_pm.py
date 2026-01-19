# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 11:04:04 2025

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
#%% PB processing pipeline


experiment_dirs = [
   # r'Y:\Data\FCI\Hedwig\PFNpm_SS00191\251210\f1\Trial1', # Not amazing behaviour but a good start for analysis,
                r'Y:\Data\FCI\Hedwig\PFNpm_SS52245\260108\f1\Trial1',
                r'Y:\Data\FCI\Hedwig\PFNpm_SS52245\260108\f1\Trial2',
                r'Y:\Data\FCI\Hedwig\PFNpm_SS52245\260108\f1\Trial5'
                   ]
regions = ['PFN_L','PFN_R']
regions2 = ['pfn_l_16','pfn_r_16']
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
    
    # Upsample columnar data from 8 to 16 columns
    cx.upsample8_2_16(regions)
    
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()#upsample to 50Hz
    pv2, ft, ft2, ix = cx.load_postprocessing()
    cxa = CX_a(datadir,regions=regions2,yoking=False)
    
    
        
    
    cxa.save_phases()
    
#%%
datadir = r'Y:\Data\FCI\Hedwig\PFNpm_SS52245\260108\f1\Trial2'
regions2 = ['pfn_l_16','pfn_r_16'] 
cxa = CX_a(datadir,regions2,denovo=False)
cxa.simple_raw_plot(regions = regions2,yeseb=False)
#cxa.simple_raw_plot(plotphase=True,regions = regions2,yeseb=False)
#%% FSB processing pipeline
experiment_dirs = [
                r'Y:\Data\FCI\Hedwig\PFNpm_SS52245\260108\f1\Trial3',
                r'Y:\Data\FCI\Hedwig\PFNpm_SS52245\260108\f1\Trial4',
                   ]
regions = ['fsb']

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
    cxa = CX_a(datadir,regions=regions,yoking=False)

    cxa.save_phases()

#%%
datadir = r'Y:\Data\FCI\Hedwig\PFNpm_SS52245\260108\f1\Trial4'
regions2 = ['fsb'] 
cxa = CX_a(datadir,regions2,denovo=False)
cxa.simple_raw_plot(regions = regions2,yeseb=False)
cxa.simple_raw_plot(plotphase=True,regions = regions2,yeseb=False)

#%% 
ampL = np.mean(cxa.pdat['wedges_pfn_l_16'],axis=1)
ampR = np.mean(cxa.pdat['wedges_pfn_r_16'],axis=1)
#ampL = np.max(cxa.pdat['wedges_pfn_l_16'],axis=1)
#ampR = np.max(cxa.pdat['wedges_pfn_r_16'],axis=1)
ampL = (ampL-np.mean(ampL))/np.std(ampL)
ampR = (ampR-np.mean(ampR))/np.std(ampR)
heading = cxa.ft2['ft_heading'].to_numpy()
plt.figure()
plt.plot(ampL,color='b')
plt.plot(ampR,color='r')
d = (ampR-ampL)/(ampL)
d = d-np.mean(d)
plt.plot(d)
hL = heading<0
plt.plot(heading,color='k')
plt.figure()
plt.subplot(1,2,1)
plt.hist(ampL[hL],bins=np.linspace(-2,2,50),alpha=.5)
plt.hist(ampL[~hL],bins=np.linspace(-2,2,50),alpha=.5)


plt.subplot(1,2,2)
plt.hist(ampR[hL],bins=np.linspace(-2,2,50),alpha=.5)
plt.hist(ampR[~hL],bins=np.linspace(-2,2,50),alpha=.5)


#%%

x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
t = cxa.pv2['relative_time'].to_numpy()
x,y= ug.fictrac_repair(x,y)
u = ug()
vx,vy,vd = u.get_velocity(x,y,t)
plt.plot(vd)
va = ug.get_ang_velocity(heading,t)

#%% 
plt.plot(ampL,color='b')
plt.plot(ampR,color='r')
#plt.plot(vx/10,color=[0.6,0.6,0.6])
#plt.plot(vy/10,color='k')
plt.plot(vd/10,color='g')
ins = cxa.ft2['instrip'].to_numpy()
plt.plot(ins,color='r')
plt.plot(np.abs(va)/10,color='k')
#%%
plt.close('all')
heading = cxa.ft2['ft_heading'].to_numpy()

L = cxa.pdat['phase_pfn_l_16']
R = cxa.pdat['phase_pfn_r_16']

plt.figure()
plt.subplot(1,2,1)
plt.scatter(heading,L,color='k',s=1)
plt.title('Left')

plt.subplot(1,2,2)
plt.scatter(heading,R,color='k',s=1)
plt.title('Right')

plt.figure()
plt.plot(heading,color='k')
plt.plot(L,color='b')
plt.plot(R,color='r')
s = 180*ug.circ_subtract(L,np.squeeze(R))/np.pi
180*stats.circmean(ug.circ_subtract(L,np.squeeze(R)),high=np.pi,low=-np.pi)/np.pi
#%%
plt.plot(s)
plt.plot(ampL*20,color='b')
plt.plot(ampR*20,color='r')
plt.plot([0,len(s)],[-45,-45],color='k')
#%% Notes
# PFNpc gain seems modulated by forward velocity (odour?) and turn direction
# This change in the signal with odour could be wind speed. I think it could be good
# To replace the odour vial with water and see what happens. The reduced volume in the vial may be allowing wind speed to increase