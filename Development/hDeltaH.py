# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 15:53:36 2025

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
experiment_dirs = [
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial1",# Ramping like FB6H, cross-wind encoding to left. Could be issue with recording
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial2",# Varied recovery with ET and plume distance
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial3", # messed up pulse experiment
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial4",# ACV pulses
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial5",# Oct pulses
    
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f2\Trial2",#Running through plumes, recovery matches distance from plume well
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f2\Trial3", Long returns and plume traversals, fast recovery dynamics
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f2\Trial4", #Oct pulses
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f2\Trial5",# ACV pulses
    
    # r'Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251209\f1\Trial1', # Ok behaviour
    # r'Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251209\f1\Trial2',#Running through plumes, v rapid recovery
    # r'Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251209\f1\Trial3', # Ok tracking and behaviour
    
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f1\Trial1", #Decent tracking
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f1\Trial2", # Decent tracking, some traversing also
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f1\Trial3", # Not much tracking downwind walking, v rapid recovery of activity
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f1\Trial4",#Oct pulses
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f1\Trial5"# ACV pulses
    
    
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f2\Trial1", #Nice dataset. Some ET bouts followed by plume traversal
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f2\Trial2", # Decent tracking, some traversing also
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f2\Trial3", # Lots of tight exploration but little plume tracking, maybe this fly was unlucky
    # r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f2\Trial4",#ACV pulses
    r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f2\Trial5"# Oct pulses
    
                   ]

regions = ['eb','fsb_upper','fsb_lower']
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
    cxa = CX_a(datadir,regions=regions)
    
    cxa.save_phases()
#%% 
plt.close('all')
datadir = r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f2\Trial1"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
cxa.simple_raw_plot(plotphase=False,regions = ['eb','fsb_upper','fsb_lower'],yk='eb')

cxa.simple_raw_plot(plotphase=True,regions = ['eb','fsb_upper','fsb_lower'],yk='eb')

cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 4)
#%% 
plt.figure()
ins = cxa.ft2['instrip'].to_numpy()
amp = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
#amp = np.max(cxa.pdat['wedges_fsb_upper'],axis=1)
plt.plot(amp)
plt.plot(ins,color='r')
jumps = cxa.get_entries_exits_like_jumps()
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
dx = np.append(0,np.diff(x))
dy = np.append(0,np.diff(y))
sl_dist = np.zeros(len(x))
for j in jumps:
    dx = np.arange(j[1],j[2])
    ty = y[dx]
    tx = x[dx]
    ty = ty-ty[0]
    tx = tx-tx[0]
    sl_dist[dx] = np.sqrt(np.sqrt(tx**2+ty**2))
dx = np.arange(jumps[-1,2],len(amp))
ty = y[dx]
tx = x[dx]
ty = ty-ty[0]
tx = tx-tx[0]
sl_dist[dx] = np.sqrt(np.sqrt(tx**2+ty**2))
plt.plot(sl_dist/20,color='k')

fci = fci_regmodel(amp,cxa.ft2,cxa.pv2)
ampz = (amp-np.mean(amp))/np.std(amp)
plt.figure()
fci.example_trajectory_scatter(ampz,cmin=-2,cmax=2)
#%% load data
datadirs = [r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f1\Trial2",
    r'Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251209\f1\Trial1',
    r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f1\Trial2",
            r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\260109\f2\Trial1"]
data = {}
for i,d in enumerate(datadirs):
    cxa = CX_a(d,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    data.update({str(i):cxa})
#%%
plt.close('all')
for d in data:
     cxa = data[d]
     
     # Plot mean fluorescence across columns lined up to first entry
     plt.figure(1)
     amp = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
     t = np.arange(0,len(amp))/10
     tstart = np.where(cxa.ft2['instrip'].to_numpy())[0][0]
     t = t-t[tstart]
     plt.plot(t,amp)
     
     plt.figure()
     eb = cxa.pdat['phase_eb']
     ebs= ug.savgol_circ(eb,500,3)
     fsb = cxa.pdat['phase_fsb_upper']
     plt.scatter(t,eb,color='k',s=2)
     plt.plot(t,ebs,color=[0.5,0.5,0.5])
     plt.scatter(t,fsb,color='b',s=2)
     plt.plot(t,amp*10-12)
     u = ug()
     vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
     vd = vd/np.std(vd)
     vd = np.append(0,vd)
     plt.plot(t,vd-12,color='k')
     
     plt.figure()
     plt.scatter(cxa.pdat['offset_eb_phase'],cxa.pdat['offset_fsb_upper_phase'],s=3)
     
#%% 
fsb_phase = cxa.pdat['phase_fsb_upper']
eb_phase = cxa.pdat['phase_eb']
u = ug()
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
t= cxa.pv2['relative_time'].to_numpy()
vx,vy,vd = u.get_velocity(x,y,t)
vd = np.append(0,vd)
still,stillsize = ug.find_blocks(vd<0.2)
sdx = np.zeros(len(vd),dtype=bool)
for si,s in enumerate(still):
    dx = np.arange(s,s+stillsize[si])
    if len(dx)>20:
        print(si)
        sdx[dx[20:]] =True
plt.scatter(eb_phase[vd>0.5],fsb_phase[vd>0.5],s=2,color='k')
plt.scatter(eb_phase[sdx],fsb_phase[sdx],s=2,color='r')

#%%
  # if set(['train_heading']).issubset(self.ft2):
  #   plt.figure()
#cxa.mean_jump_arrows(fsb_names=['fsb_upper'],ascale=100,jsize=5)
#plt.ylim([-40,40])
idx = np.arange(8400,10100)
cxa.plot_traj_arrow_segment(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/5,axis=1),idx,a_sep= 2)