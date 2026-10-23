# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:44:09 2026

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
import pingouin as pg
regions = ['eb' ,'fsb_upper','fsb_lower']

#%%
experiment_dirs = [
  r'Y:\Data\FCI\Hedwig\hDeltaJ_Gr64f_EPG_RCaMP\261009\f1\Trial1',
  r'Y:\Data\FCI\Hedwig\hDeltaJ_Gr64f_EPG_RCaMP\261009\f1\Trial2' # Maybe some enhanced inhibition.
          
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
    cxa.simple_raw_plot(regions=['fsb_upper','fsb_lower'],plotphase=True)
    
    
#%% Check data


#datadir = r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260824\f1\Trial2'

datadir=  r'Y:\Data\FCI\Hedwig\hDeltaJ_Gr64f_EPG_RCaMP\261009\f1\Trial2'
cxa = CX_a(datadir,regions=regions,yoking=True,stim=True,denovo=False)
plt.close('all')
amp = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
#amp = ug.get_pvas(cxa.pdat['wedges_fsb_upper'])
cxa.simple_raw_plot(regions=['eb','fsb_upper','fsb_lower'],plotphase=False)

plt.figure()
ins = cxa.ft2['instrip'].to_numpy()>0
led = cxa.ft2['led1_stpt'].to_numpy()==0
odour = np.logical_and(ins,~led)
ledsin = np.logical_and(led,~ins)
both = np.logical_and(ins,led)
t = np.arange(0,len(amp))/10
plt.plot(t,amp)
plt.plot(t,odour)
plt.plot(t,both)
plt.plot(t,ledsin)
x = cxa.ft2['ft_posx'].to_numpy();y = cxa.ft2['ft_posx'].to_numpy()
u = ug()
_,_,dd = u.get_velocity( x, y, t)
plt.plot(t[1:],-1+dd/20 ,color='k')


timint = 300

ob,obs = ug.find_blocks(odour)
odata = np.zeros((timint,len(ob)))*np.nan
for i,o in enumerate(ob):
    dx = np.arange(o,min(o+timint,len(amp)))
    dxmn = np.arange(o-5,o)
    odata[:len(dx),i] = amp[dx] #- np.mean(amp[dxmn])
    
ob,obs = ug.find_blocks(ledsin)
ldata = np.zeros((timint,len(ob)))*np.nan
for i,o in enumerate(ob):
    dx = np.arange(o,min(o+timint,len(amp)))
    dxmn = np.arange(o-5,o)
    ldata[:len(dx),i] = amp[dx] #- np.mean(amp[dxmn])
    
    
ob,obs = ug.find_blocks(both)
bdata = np.zeros((timint,len(ob)))*np.nan
for i,o in enumerate(ob):
    dx = np.arange(o,min(o+timint,len(amp)))
    dxmn = np.arange(o-5,o)
    bdata[:len(dx),i] = amp[dx]#- np.mean(amp[dxmn])
    
    
omn = np.nanmean(odata,axis=1)
lmn = np.nanmean(ldata,axis=1)
bmn = np.nanmean(bdata,axis=1)
plt.figure()
t = np.arange(0,timint)/10;t = t-2
plt.plot(t,omn,color='k',label='ACV')
plt.plot(t,lmn,color='r',label='LED')
plt.plot(t,bmn,color='m',label='ACV + LED')
plt.fill([0,-2,-2,0],[-.5,-.5,.5,.5],color=[.8,.8,.8],linewidth=0)
plt.legend()
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 5)
cxa.plot_traj_arrow_heat(['fsb_upper'], amp,cmin=0,cmax=.6,colormap='inferno')
