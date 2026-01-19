# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 18:04:31 2026

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


#%%
experiment_dirs = [
    #r'Y:\Data\FCI\Hedwig\ChAT\260102\f1\Trial2',
                   r'Y:\Data\FCI\Hedwig\ChAT\260102\f1\Trial3'
   
                   ]
regions = ['eb','fsb_upper','fsb_middle','fsb_lower']
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
datadir =r'Y:\Data\FCI\Hedwig\ChAT\260102\f1\Trial2'
regions = ['eb','fsb_upper','fsb_middle','fsb_lower']

cxa = CX_a(datadir,regions=regions,denovo=False)
# %% Amplitude across experiment
regions = ['eb','fsb_lower','fsb_middle','fsb_upper']
colours = np.array([[0,0,0],[81,156,205],[49,99,125],[81,61,204]])/255
ins = cxa.ft2['instrip'].to_numpy().astype(float)
x = np.arange(0,len(ins))/10
plt.fill_between(x,ins*0,ins*3,color='r',zorder=0,linewidth=0)
offset = 0
for ir,r in enumerate(regions):
    amp = np.mean(cxa.pdat['wedges_'+r],axis=1)
    plt.plot(x,amp+offset,color=colours[ir,:],label=r)
    offset+=.5
u = ug()
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
plt.plot(x[1:],vd/20 -1,color=[0.5,0.5,.5])
plt.legend()
#%% Plot on trajectory
r = 'fsb_upper'
ca = np.mean(cxa.pdat['wedges_'+r],axis=1)
ca = (ca-np.mean(ca))/np.std(ca)
fci = fci_regmodel(ca.copy(),cxa.ft2,cxa.pv2)
#fci.example_trajectory_scatter(ca.copy(),cmin=-2,cmax=2)
fci.example_trajectory_jump(ca.copy(),cxa.ft,cmin=-2,cmax=2)
# %% Phase across experiment
regions = ['eb','fsb_lower','fsb_middle','fsb_upper']
colours = np.array([[0,0,0],[81,156,205],[49,99,125],[81,61,204]])/255
ins = cxa.ft2['instrip'].to_numpy().astype(float)
x = np.arange(0,len(ins))/10
plt.fill_between(x,ins*0-np.pi,ins*2*np.pi-np.pi,color='r',zorder=0,linewidth=0)
offset = 0
for ir,r in enumerate(regions):
    p = cxa.pdat['offset_'+r+'_phase'].to_numpy()
    plt.scatter(x,p,color=colours[ir,:],s=2,label=r)
plt.legend()
    
#%%
plt.close('all')

cxa.simple_raw_plot(plotphase=False,regions = ['eb','fsb_lower','fsb_middle','fsb_upper'],yk='eb')
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_middle_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_middle']/2,axis=1),a_sep= 2)
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_lower_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_lower']/2,axis=1),a_sep= 2)


cxa.simple_raw_plot(plotphase=True,regions = ['eb','fsb_upper','fsb_middle','fsb_lower'],yk='eb')

# plt.figure()
# wedges = cxa.pdat['wedges_fsb_upper']
# wamp = np.mean(wedges,axis=1)
# plt.plot(wamp,color='k')
# plt.plot(cxa.ft2['mfc3_stpt']/np.max(cxa.ft2['mfc3_stpt']),color='g')
# plt.plot(cxa.ft2['instrip'],color='r')

# try :
#     plt.figure()
#     cxa.mean_jump_arrows()
#     cxa.mean_jump_lines()
# except:
#     print('no jumps')
#%% 
plt.close('all')
for ir,r in enumerate(regions):
    plt.subplot(2,2,ir+1)
    plt.scatter(180*cxa.ft2['ft_heading']/np.pi,180*cxa.pdat['offset_' +r+'_phase']/np.pi,color=colours[ir,:],s=2,alpha=0.3)
    plt.xticks(np.arange(-180,181,90))
    plt.yticks(np.arange(-180,181,90))
    plt.ylabel('phase '+r+' (deg)')
    plt.xlabel('heading (deg)')
    plt.ylim([-180,180])
    plt.xlim([-180,180])
#%%
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_lower_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_lower']/2,axis=1),a_sep= 5)






