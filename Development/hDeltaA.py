# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:03:50 2024

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

for i in [1,2,3]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\hDeltaA_GluSnFR\250812\f2\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
#%% Glu Snfr
datadir =r"Y:\Data\FCI\Hedwig\hDeltaA_GluSnFR\250812\f2\Trial3"
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]

cx = CX(name,['fsb'],datadir)

# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()#upsample to 50Hz
pv2, ft, ft2, ix = cx.load_postprocessing()
cxa = CX_a(datadir,regions=['fsb'],yoking=False)
plt.figure()
cxa.simple_raw_plot(plotphase=True,regions = ['fsb'],yeseb=False) 
cxa.save_phases()    
#%% Check GluSnfr
datadir =r"Y:\Data\FCI\Hedwig\hDeltaA_GluSnFR\250812\f2\Trial2"
cxa = CX_a(datadir,regions=['fsb'],yoking=False,denovo=False)
cxa.simple_raw_plot(plotphase=True,regions = ['fsb'],yeseb=False) 

phase = cxa.pdat['phase_fsb']
heading = cxa.ft2['ft_heading'].to_numpy()
ins = cxa.ft2['instrip'].to_numpy()
t = np.arange(0,len(heading))/10

plt.figure()
plt.plot(t,heading,color='k')
plt.plot(t,ins*np.pi,color='r')
#plt.scatter(t,phase,color='b',s=2)
# psmooth = ug.boxcar_circ(phase,20)
# plt.plot(t,psmooth,color='g')
psmooth = ug.savgol_circ(phase,20,3)
plt.scatter(t,psmooth,color='g',s=2)

plt.figure()
plt.scatter(psmooth,heading,color='k',s=1)
#%% To check
experiment_dirs = [
    # "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241120\\f1\\Trial1",
    # "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241120\\f2\\Trial1",
    # "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241120\\f2\\Trial2",
    
    "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f1\\Trial1",#2 jumps
    "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f1\\Trial2",#
    "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f1\\Trial3",
    "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f2\\Trial1",#5 jumps
    "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f2\\Trial2",
    
    
                   
                   ]
regions = ['fsb_lower','fsb_upper','eb']
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
    cxa = CX_a(datadir,regions=np.flipud(regions))
    plt.figure()
    cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb') 
    cxa.save_phases()
    
    
#%%
for e in experiment_dirs:
    datadir =os.path.join(e)
    cxa = CX_a(datadir,regions=np.flipud(regions),denovo='False')
    cxa.simple_raw_plot(plotphase=True,regions = ['fsb_upper','fsb_lower'],yk='eb') 
    cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),cxa.pdat['amp_fsb_upper'],a_sep= 5)

#%%
datadir=  r"Y:\Data\FCI\Hedwig\hDeltaA_SS64464\\241121\f2\Trial1"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
#%%
savedir = r'Y:\Data\FCI\FCI_summaries\hDeltaA'
cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb')
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_lower_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_lower']/2,axis=1),a_sep= 2)

plt.figure()
t  = np.arange(0,len(cxa.pdat['phase_eb']))/10
plt.scatter(t,cxa.pdat['phase_eb'],color='k',s=2)
plt.scatter(t,cxa.pdat['phase_fsb_upper'],color='b',s=2)
plt.scatter(t,cxa.pdat['phase_fsb_lower'],color='m',s=2)
plt.plot(t,cxa.ft2['instrip']*3,color='r')

plt.figure()
cd = np.abs(ug.circ_subtract(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_eb']))
cd2 = np.abs(ug.circ_subtract(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_fsb_lower']))
plt.plot(t,cd,color='b')
#plt.plot(t,cd2,color='m')
plt.plot(t,cxa.ft2['instrip'],color='r')
plt.figure()
cxa.mean_jump_arrows(fsb_names=['fsb_upper','fsb_lower'],jsize =3,ascale=100)

#%%
plt.figure()
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],linestyle='--',color='r')
plt.plot([-np.pi,0],[0,np.pi],linestyle='--',color='r')
plt.plot([0,np.pi],[-np.pi,0],linestyle='--',color='r')

plt.scatter(cxa.pdat['phase_eb'],cxa.pdat['phase_fsb_upper'],color='k',s=1,alpha=.2)
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.xlabel('EPG phase')
plt.ylabel('FSB upper phase')
plt.savefig(os.path.join(savedir,'FSB_EPG_Phase.png'))

plt.figure()
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],linestyle='--',color='r')
plt.plot([-np.pi,0],[0,np.pi],linestyle='--',color='r')
plt.plot([0,np.pi],[-np.pi,0],linestyle='--',color='r')
plt.scatter(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_fsb_lower'],color='k',s=1,alpha=.2)
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.xlabel('FSB upper phase')
plt.ylabel('FSB lower phase')
plt.savefig(os.path.join(savedir,'FSB_upper_lower_phase.png'))
#%%
x = np.arange(0,len(cxa.pdat['phase_fsb_upper']))/10
plt.scatter(x,cxa.pdat['phase_fsb_upper'],color='b',s=5)
#plt.scatter(x,cxa.pdat['phase_eb'],color='k',s=2)
eb180 = cxa.pdat['phase_eb']
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