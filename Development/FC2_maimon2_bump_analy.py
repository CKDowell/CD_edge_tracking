# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:00:18 2025

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
from analysis_funs.CX_behaviour_pred_col import CX_b
plt.rcParams['pdf.fonttype'] = 42 

#%% 
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial5",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial2",
r"Y:\Data\FCI\Hedwig\FC2_maimon2\250128\f1\Trial4"]
#%%

# for datadir in datadirs:

datadir = datadirs[5]
angles = np.linspace(-np.pi,np.pi,16)

d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)

#y = cxa.pdat['amp_fsb_upper']

# pva


weds = np.sum(cxa.pdat['wedges_fsb_upper']*np.sin(angles),axis=1)
wedc = np.sum(cxa.pdat['wedges_fsb_upper']*np.cos(angles),axis=1)
pva  = np.sqrt(weds**2+wedc**2)
p0 = np.mean(pva[pva<np.percentile(pva,10)])
pva = (pva-p0)/p0

# pva_norm - measure of coherence

wednorm = cxa.pdat['wedges_fsb_upper']
wednorm = wednorm/np.max(wednorm,axis=1)[:,np.newaxis]

weds = np.sum(wednorm*np.sin(angles),axis=1)
wedc = np.sum(wednorm*np.cos(angles),axis=1)
pva_norm  = np.sqrt(weds**2+wedc**2)
p0 = np.mean(pva_norm[pva_norm<np.percentile(pva_norm,10)])
pva_norm = (pva_norm-p0)/p0



ymn = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
y0 = np.mean(ymn[ymn<np.percentile(ymn,10)])
ymn = (ymn-y0)/y0
pva_z = pva/np.std(pva)
pvan_z = pva_norm/np.std(pva_norm)
ymn_z = ymn/np.std(ymn)
ft2 = cxa.ft2
pv2 = cxa.pv2
fci = fci_regmodel(pva_norm,ft2,pv2)
plt.figure()
fci.example_trajectory_jump(cmin=0,cmax=6)

# plt.plot(pva_z)
# plt.plot(ymn_z)
# plt.plot(pva_norm)
#%%
datadir = datadirs[5]
cxb = CX_b(datadir,regions = ['eb','fsb_upper','fsb_lower'])
cxb.reg_traj_model(twindow=2,regions=['fsb_lower'])

#%% 
fc = fci_regmodel(pva_norm,ft2,pv2)
fc.rebaseline()
fc.example_trajectory_jump(cmin=-2,cmax=2)

#%% cross correlation
from scipy import signal as sg


c = sg.correlate(pva_z,ymn_z,mode='same')
#plt.plot(c)

def time_varying_correlation(x,y,window):
    iter = len(x)-window
    output = np.zeros(len(x))
    for i in range(iter):
        idx = np.arange(i,i+window)
        cor = np.corrcoef(x[idx],y[idx])
        
        output[i+window] = cor[0,1] 
    return output

#t_c = time_varying_correlation(pva_z, ymn_z, 20)
#plt.plot(t_c)

plt.plot(pvan_z)
plt.plot(cxa.ft2['instrip'],color='k')

def sine_correlation(wedge,phase):
    output = np.zeros(len(phase))
    angles = np.linspace(-np.pi,np.pi,16)
    for i,p in enumerate(phase):
        tfit = np.cos(angles-p)
        cor = np.corrcoef(wedge[i,:],tfit)
        output[i] = cor[0,1]
        
    return output
#%%         

s_c_fsb = sine_correlation(cxa.pdat['wedges_fsb_upper'],cxa.pdat['phase_fsb_upper'])
plt.plot(s_c_fsb,color='b')

s_c = sine_correlation(cxa.pdat['wedges_eb'],cxa.pdat['phase_eb'])
plt.plot(s_c,color='k')
plt.plot(ft2['instrip'],color='r')
#%%
t_c = time_varying_correlation(pvan_z, ymn_z, 10)
ft2 = cxa.ft2
pv2 = cxa.pv2
fc = fci_regmodel(s_c_fsb,ft2,pv2)
#fc.rebaseline(plotfig=True)
fc.example_trajectory_jump(s_c_fsb,cxa.ft,cmin=0.25,cmax=1) # plot with phase on top
#%%
plt.close('all')
jumps = cxa.get_jumps()
for j in jumps:
    plt.figure()
    ip = np.arange(j[0],j[1]+1)
    op = np.arange(j[1],j[2])
    plt.plot(s_c_fsb[ip],ymn_z[ip]-ymn_z[ip[0]],color='r')
    #plt.scatter(pvan_z[ip[0]],pvan_z[ip[0]],color='r')
    plt.plot(s_c_fsb[op],ymn_z[op]-ymn_z[ip[0]],color='k')
    plt.xlabel('PAV norm')
    plt.ylabel('Mean Fluor')
    plt.xlim([-1,1])
    plt.ylim([-4,4])
