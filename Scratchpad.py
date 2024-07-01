# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:27:58 2024

@author: dowel
"""

from analysis_funs.CX_imaging import CX
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
from src.utilities import funcs as fn
import pickle
from scipy.optimize import curve_fit
from analysis_funs.CX_phase_modelling import CX_phase_modelling
from analysis_funs.CX_analysis_col import CX_a
#%%
datadir = "Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
#
#%%
cxp = CX_phase_modelling(cxa)
phase = cxa.pdat['phase_eb']

x= np.ones((2,len(phase)))
x[0,:] = cxa.ft2['ft_heading']
w1 = 1 
w2 = 2
weights = np.array([w1,w2])

#cxp.phase_function(x,w1,w2)

cxp.fit_phase_function(x,phase)

plt.scatter(x[0,:],phase)
plt.scatter(x[0,:],cxp.phase_function(x,*cxp.popt))
#%%
plt.close('all')
phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
#phase = cxa.pdat['offset_eb_phase']
#phase = cxa.ft2['ft_heading'].to_numpy()
x= np.zeros((3,len(phase)))
x[0,:] = cxa.pdat['offset_eb_phase']
x[1,:] = cxp.plume_memory()

cxp.fit_phase_function(x,phase)
plt.plot(phase)
plt.plot(cxp.phase_function(x,*cxp.results.x))
plt.figure()
plt.scatter(phase,cxp.phase_function(x,*cxp.results.x),s=1)
#%%
plt.close('all')
x= np.zeros((3,len(phase)))
x[0,:] = cxa.pdat['offset_eb_phase']
x[1,:] = cxp.plume_memory()
parts = ['Pre Air','Returns','Jump Returns','In Plume']
cxp.fit_in_parts(x,phase,parts)
#cxp.reg_in_parts(x,phase,['Pre Air','Returns','Jump Returns','In Plume'])
popt_array = cxp.popt_array
popt_array = popt_array/np.expand_dims(np.max(np.abs(popt_array),axis=0),axis=0)
plt.figure()
plt.plot(popt_array)
plt.legend(parts)
#%%
phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
plt.close('all')
phase_eb = cxa.pdat['offset_eb_phase']
#phase_eb = cxa.ft2['ft_heading'].to_numpy()
parts = ['Pre Air','Returns','Jump Returns','In Plume']
dx = cxp.output_time_epochs(cxa.ft2,'Jump Returns')
xm = cxp.plume_memory()

ddiff = np.diff(dx)
e_end = np.where(ddiff>1)[0]
e_end = np.append(e_end,len(ddiff)-1)
estart = np.append(1,e_end[:-1]+1)
endx = dx[e_end]
stdx = dx[estart]
times = cxa.pv2['relative_time'].to_numpy()
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
x,y = cxa.fictrac_repair(x,y)
amp = cxa.pdat['amp_fsb_upper']
pamp = np.percentile(amp,99)
amp[amp>0.2] = 0.2 
amp = amp/0.2

p_scat = np.zeros((len(endx),2))
for i,e in enumerate(endx):

    tdc = np.arange(stdx[i],e,1,dtype=int)
    t = times[tdc]
    t = t-t[0]
    tx = x[tdc]
    ty = y[tdc]
    tx = tx-tx[0]
    ty = ty-ty[0]
    tphase = phase[tdc]
    d = np.sqrt(tx**2+ty**2)
    
    ttdx = (t-max(t))>-1
    p_scat[i,0] = circmean(tphase[ttdx],high=np.pi,low=-np.pi)
    p_scat[i,1] = xm[tdc[-1]]
    # plt.figure()
    # plt.plot(tx,ty)
    # ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(tphase, t,color='k')
    #ax.plot(phase[tdc],t,color=[0.2,0.2,0.8])
    ax.scatter(phase[tdc],t,s=amp[tdc]*100,color=[0.2,0.2,0.8],alpha=amp[tdc])
    ax.plot(xm[tdc],t,color='r')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    #ax.set_ylim([0,60])
    # plt.figure()
    # plt.plot(phase_eb[tdc],phase[tdc],color='k')
    # plt.scatter(phase_eb[tdc],phase[tdc],c=tdc-tdc[0],s=10,zorder=10)
    # plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k')
    # plt.scatter(phase_eb[tdc[0]],x[1,tdc[0]],color='r',zorder=11)
plt.figure()
plt.scatter(p_scat[:,1],p_scat[:,0])
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.plot([0,0],[-np.pi,np.pi],color='k')
plt.plot([-np.pi,np.pi],[0,0],color='k')
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
#%% Scatter of last phase before re-entry vs remembered angle
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2"]
savedir= "Y:\Data\FCI\FCI_summaries\FC2_maimon2"
plt.close('all')
plt.figure()
colours = np.array([[166,206,227],
[202,178,214],
[51,160,44],
[251,154,153],
[227,26,28],
[253,191,111],
[255,127,0],
[31,120,180],
[178,223,138]
])/255
for i,datadir in enumerate(datadirs):
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    cxp = CX_phase_modelling(cxa)
    phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    p_scat = cxp.phase_memory_scatter(phase)
    p_smn = circmean(p_scat,high=np.pi,low=-np.pi,axis=0)
    p_scat = 180*p_scat/np.pi
    p_smn = 180*p_smn/np.pi
    plt.scatter(p_scat[:,1],p_scat[:,0],s=20,color=colours[i,:],zorder=9,alpha=0.5)
    plt.scatter(p_smn[1],p_smn[0],marker='+',s=200,color=colours[i,:],zorder=10)
plt.xlim([-180,180])
plt.ylim([-180,180])
plt.plot([0,0],[-180,180],color='k')
plt.plot([-180,180],[0,0],color='k')
plt.plot([-180,180],[-180,180],color='k',linestyle='--')
plt.plot([-180/2,-180/2],[-180,0],color='r',linestyle='--')
plt.plot([-180,0],[-180/2,-180/2],color='r',linestyle='--')
plt.xlabel('Prior return angle (deg)')
plt.ylabel('FC2 phase 2s return (deg)')
plt.xticks(np.arange(-180,270,90))
plt.yticks(np.arange(-180,270,90))
plt.savefig(os.path.join(savedir,'Scatter_returnVsMemFC2.png'))
#%%
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2"]
plt.close('all')
x_offset = 0
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

for i,datadir in enumerate(datadirs):
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    cxp = CX_phase_modelling(cxa)
    phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    
    if i==0:
        pltmean = cxp.mean_phase_polar(phase,succession=[fig,ax])
        pltmean = np.expand_dims(pltmean,2)
    else:
        pm = cxp.mean_phase_polar(phase,succession=[fig,ax])
        pm = np.expand_dims(pm,2)
        pltmean = np.append(pltmean,pm,2)

    
#%%
from scipy.stats import circmean, circstd

pltmn = circmean(pltmean,high=np.pi, low=-np.pi,axis=2)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
tstandard = np.linspace(0,49,50)
colours = np.array([[0.2,0.2,0.8],[0, 0, 0],[1,0,0]])
for i in range(3):
    for i2 in range(np.shape(pltmean)[2]):
        ax.plot(pltmean[:,i,i2],tstandard,color=colours[i,:],alpha=0.25) 
        

for i in range(3):
    ax.plot(pltmn[:,i],tstandard,color=colours[i,:],linewidth=2) 
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticklabels([0,45,90,135,180,-135,-90,-45])
ax.set_title('FC2 Jump Returns')
#%% 
datadir = "Y:\\Data\\FCI\\Hedwig\\hDeltaJ\\240529\\f1\\Trial3"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
cxp = CX_phase_modelling(cxa)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
pltmean = cxp.mean_phase_polar(phase,succession=[fig,ax])
ax.set_xticklabels([0,45,90,135,180,-135,-90,-45])
ax.set_title('hDeltaJ Jump Returns')
#%% 
et_dir = [-1,-1,1,1,1,-1]
rootdir = 'Y:\\Data\\FCI\\AndyData\\hDeltaC_imaging\\csv'
folders = ['20220517_hdc_split_60d05_sytgcamp7f',
 '20220627_hdc_split_Fly1',
 '20220627_hdc_split_Fly2',
 '20220628_HDC_sytjGCaMP7f_Fly1',
 #'20220628_HDC_sytjGCaMP7f_Fly1_45-004', 45 degree plume
 '20220629_HDC_split_sytjGCaMP7f_Fly1',
 '20220629_HDC_split_sytjGCaMP7f_Fly3']
plt.close('all')
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
for i,f in enumerate(folders):
    datadir = os.path.join(rootdir,f,"et")
    cxa = CX_a(datadir,Andy='hDeltaC')
    
    cxp = CX_phase_modelling(cxa)
    cxp.side = et_dir[i]
    phase = cxa.pdat['offset_fsb_phase']
    if i==0:
        pltmean = cxp.mean_phase_polar(phase,succession=[fig,ax],part='Returns')
        pltmean = np.expand_dims(pltmean,2)
    else:
        pm = cxp.mean_phase_polar(phase,succession=[fig,ax],part='Returns')
        pm = np.expand_dims(pm,2)
        pltmean = np.append(pltmean,pm,2)
#%%
pltmn = circmean(pltmean,high=np.pi, low=-np.pi,axis=2)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
tstandard = np.linspace(0,49,50)
colours = np.array([[0.2,0.2,0.8],[0, 0, 0],[1,0,0]])
for i in range(3):
    for i2 in range(np.shape(pltmean)[2]):
        ax.plot(pltmean[:,i,i2],tstandard,color=colours[i,:],alpha=0.25) 
        

for i in range(3):
    ax.plot(pltmn[:,i],tstandard,color=colours[i,:],linewidth=2) 
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticklabels([0,45,90,135,180,-135,-90,-45])
ax.set_title('hDeltaC Returns')