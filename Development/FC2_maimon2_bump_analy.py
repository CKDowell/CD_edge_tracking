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
from Utilities.utils_general import utils_general as ug
from Utilities.utils_plotting import uplt as up
u = ug()
plt.rcParams['pdf.fonttype'] = 42 

#%% Load all data
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial5",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial2",
r"Y:\Data\FCI\Hedwig\FC2_maimon2\250128\f1\Trial4"]

all_flies = {}
for i,datadir in enumerate(datadirs):
    print(datadir)
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    all_flies.update({str(i):cxa})

#%% Stopping and starting analysis
mvthresh = 1 #1mm/s
minsize = 5
plt.close('all')
# Assess phase during returns to jumped plume, look at FC2- EPG phase lag/lead
for i in range(len(datadirs)):
    cxa = all_flies[str(i)]
    jumps = cxa.get_jumps()
    
    dx_dt,dy_dt,dd_dt =u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
    stills = dd_dt<mvthresh
    bst,bsz = ug.find_blocks(stills)
    still_start = bst[bsz>=minsize]+1 # add one because the velocity signal is one shorter
    still_size = bsz[bsz>=minsize]
    # plt.figure()
    
    # plt.subplot(2,2,1)
    # plt.plot(dd_dt)
    # plt.plot(stills*10)
    
    # plt.subplot(2,2,2)
    # plt.hist(dd_dt,bins=100)
    
    # plt.subplot(2,2,3)
    # plt.plot(stills)
    # plt.plot(cxa.ft2['instrip']/2)
    fsb = cxa.pdat['phase_fsb_upper']
    eb = cxa.pdat['phase_eb']
    heading = cxa.ft2['ft_heading'].to_numpy()
    w_fsb = cxa.pdat['wedges_fsb_upper']
    wmean = np.mean(w_fsb,axis=1)
    wmeanz = (wmean-np.mean(wmean))/np.std(wmean)
    pdiff = ug.circ_subtract(fsb,eb)
    pva = ug.get_pvas(w_fsb)
    pvaz = (pva-np.mean(pva))/np.std(pva)
    pdiff_vel = ug.circ_vel(pdiff,cxa.pv2['relative_time'],smooth=True,winlength=10)
    fc2_vel = ug.circ_vel(fsb,cxa.pv2['relative_time'],smooth=True,winlength=10)
    #pdiff_vel = ug.circ_vel(pdiff,cxa.pv2['relative_time'],smooth=False)
    pvcorr = ug.time_varying_correlation(pvaz,wmeanz,20)
    for j in jumps:
        t_stills = still_start[np.logical_and(still_start>j[1],still_start<j[2])]
        if len(t_stills)==0:
            plt.figure()
            
            plt.subplot(2,1,1)
            plt.title('Fly: ' + str(i))
            dx = np.arange(j[0],j[2])
            od_off= j[1]-j[0]
            t_fsb = fsb[dx]
            t_h = heading[dx]
            t_fsb = ug.circ_subtract(t_fsb,t_fsb[od_off])
            t_eb = eb[dx]
            t_eb = ug.circ_subtract(t_eb,t_eb[od_off])
            t_h = ug.circ_subtract(t_h,t_h[od_off])
            t_vel = dd_dt[dx]
            plt.plot(t_fsb,color='b')
            plt.plot(t_eb,color='k')
            plt.plot(t_h,color=[0.5,0.5,0.5])
            plt.plot(-5+t_vel/10,color='k')
            
            #plt.plot(pdiff[dx],color='b')
            #plt.plot(pdiff_vel[dx]/2,color='m')
           # plt.plot(fc2_vel[dx]/2,color='m')
            plt.plot([0,j[2]-j[0]],[0,0],color='k',linestyle='--')
            plt.ylim([-5,np.pi])
            plt.scatter(od_off,1,color='r')
            plt.subplot(2,1,2)
            plt.plot(t_vel,color='k')
            plt.scatter(od_off,5,color='r')
            plt.plot(wmeanz[dx]*10,color='g')
            plt.plot(pvaz[dx]*10,color='b')
       # plt.plot(pvcorr[dx]*20,color='m')
            # plt.scatter(j[1:],[1,1],color='r')
            # plt.scatter(t_stills,np.ones(len(t_stills)),color='k',zorder=2)

# Assess amplitude of bump during these epochs
#%% Phase nulled data
x = np.arange(0,16)
plt.close('all')
plotdata = np.zeros((16,len(datadirs),2,3))
for i in range(len(datadirs)):
    cxa = all_flies[str(i)]
    jumps = cxa.get_jumps()
    fsb = cxa.pdat['phase_fsb_upper']*-cxa.side
    eb = cxa.pdat['phase_eb']*-cxa.side
    
    w_fsb = cxa.pdat['wedges_fsb_upper']
    w_eb = cxa.pdat['wedges_eb']
    
    if cxa.side==1:
        w_fsb = np.fliplr(w_fsb)
        w_eb = np.fliplr(w_eb)
    
    fsb_null = ug.phase_nulling(w_fsb,fsb)
    eb_null = ug.phase_nulling(w_eb,eb)
    
    ipdx = np.array([],dtype='int')
    for j in jumps:
        dx = np.arange(j[0],j[1])
        ipdx = np.append(ipdx,dx)
    teb = np.mean(eb_null[ipdx,:],axis=0)
    tfsb = np.mean(fsb_null[ipdx,:],axis=0)
    plotdata[:,i,0,0] = teb
    plotdata[:,i,1,0] = tfsb
    
    plt.figure(1)
    plt.plot(x,teb,color='k')
    plt.plot(x+16,tfsb,color='b')
    plt.plot([7.5,7.5],[0,1],color='k',linestyle='--')
    plt.plot([23.5,23.5],[0,1],color='k',linestyle='--')
    
    ipdx = np.array([],dtype ='int')
    for j in jumps:
        dx = np.arange(j[1],j[2])
        if len(dx)>20:
            dx = dx[10:20]
        ipdx = np.append(ipdx,dx)
    teb = np.mean(eb_null[ipdx,:],axis=0)
    tfsb = np.mean(fsb_null[ipdx,:],axis=0)
    
    plotdata[:,i,0,1] = teb
    plotdata[:,i,1,1] = tfsb
    
    
    plt.plot(x+32,teb,color='k')
    plt.plot(x+16+32,tfsb,color='b')
    plt.plot([7.5,7.5],[0,1],color='k',linestyle='--')
    plt.plot([23.5,23.5],[0,1],color='k',linestyle='--')

    ipdx = np.array([],dtype ='int')
    for j in jumps:
        dx = np.arange(j[1],j[2])
        if len(dx)>20:
            dx = dx[5:]
        ipdx = np.append(ipdx,dx)
    teb = np.mean(eb_null[ipdx,:],axis=0)
    tfsb = np.mean(fsb_null[ipdx,:],axis=0)
    plt.figure(3)
    plt.plot(x,teb,color='k')
    plt.plot(x+16,tfsb,color='b')
    plt.plot([7.5,7.5],[0,1],color='k',linestyle='--')
    plt.plot([23.5,23.5],[0,1],color='k',linestyle='--')


    plotdata[:,i,0,2] = teb
    plotdata[:,i,1,2] = tfsb
pmean = np.mean(plotdata,axis=1)
for i in range(2):
    plt.figure()
    for j in range(3):
        y = pmean[:,i,j]
        plt.plot(x,y)
#%% Phase nulled bump progression for jump returns for individuals
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
plt.close('all')
bins = 5
fig = plt.figure()
ax = fig.add_subplot(111)
plotdata = np.zeros((16,bins*2,2,len(datadirs)))
for i in range(len(datadirs)):
    
    offset = 0
    cxa = all_flies[str(i)]
    plotdata[:,:,:,i] = cxa.phase_nulled_jump(bins=bins,fsb_names=['eb','fsb_upper'])
    
    
pltmean = np.mean(plotdata,axis=3)

cmap = plt.get_cmap('cividis')
colours = cmap(np.linspace(0,1,bins*2))

asymmetry = np.sum(np.abs(plotdata-np.flipud(plotdata)),axis=0)

for i in range(pltmean.shape[1]):
    #plt.plot(x,tdata[:,i,0],color =colours[i,:])
    #ax.plot3D(x+offset,x*0+i,tdata[:,i,0],color =colours[i,:])
    ax.plot(x+offset,pltmean[:,i,0],color ='k')
    ax.plot(x+offset,plotdata[:,i,0,:],color ='k',alpha=0.3)
    ax.plot(x+offset,pltmean[:,i,1],color ='b')
    ax.plot(x+offset,plotdata[:,i,1,:],color ='b',alpha=0.3)
    offset = offset+16

plt.figure() # more work on this
plt.plot(asymmetry[:,0,:],color='k')
plt.plot(asymmetry[:,1,:],color='b')

#%% Ca across wedges for returns
plt.close('all')

for datadir in datadirs:
#datadir = datadirs[5]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    wed_arr = cxa.mean_jump_wedges(fsb_names=['fsb_upper','eb'])
    wed_arr_mean = np.mean(wed_arr,axis=2)
    
    # plt.subplot(1,2,1)
    # plt.imshow(np.flipud(wed_arr[:,:,5,0]),interpolation=None,cmap='Blues')
    # plt.subplot(1,2,2)
    # plt.imshow(np.flipud(wed_arr[:,:,5,1]),interpolation=None,cmap='Greys')
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(np.flipud(wed_arr_mean[:,:,0]),interpolation=None,cmap='Blues',vmin=0,vmax=1)
    plt.plot([0.5,15.5],[50,50],color='r',linestyle='--')
    plt.plot([7.5,7.5],[0,100],color='w',linestyle='--')
    if cxa.side==1:
        plt.plot([11.5,11.5],[0,50],color='r',linestyle='--')
    else:
        plt.plot([3.5,3.5],[0,50],color='r',linestyle='--')
    plt.xticks([])
    plt.subplot(1,2,2)
    plt.imshow(np.flipud(wed_arr_mean[:,:,1]),interpolation=None,cmap='Greys',vmin=0,vmax=1)
    plt.plot([0.5,15.5],[50,50],color='r',linestyle='--')
    plt.xticks([])
    plt.plot([0.5,15.5],[50,50],color='r',linestyle='--')
    plt.plot([7.5,7.5],[0,100],color='w',linestyle='--')
    if cxa.side==1:
        plt.plot([11.5,11.5],[0,50],color='r',linestyle='--')
    else:
        plt.plot([3.5,3.5],[0,50],color='r',linestyle='--')
        
    
    wed_rand,pon,phases = cxa.random_jump_wedge(fsb_names=['fsb_upper','eb'])
    wlen = wed_rand.shape[0]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(np.flipud(wed_rand[:,:,0]),interpolation=None,cmap='Blues')
    plt.plot([0.5,15.5],[wlen-pon,wlen-pon],color='r',linestyle='--')
    plt.plot([7.5,7.5],[0,wlen],color='w',linestyle='--')
    if cxa.side==1:
        plt.plot([11.5,11.5],[0,wlen-pon],color='r',linestyle='--')
    else:
        plt.plot([3.5,3.5],[0,wlen-pon],color='r',linestyle='--')
    plt.xticks([])
    plt.subplot(1,2,2)
    plt.imshow(np.flipud(wed_rand[:,:,1]),interpolation=None,cmap='Greys')
    plt.plot([0.5,15.5],[wlen-pon,wlen-pon],color='r',linestyle='--')
    plt.xticks([])
    plt.plot([0.5,15.5],[wlen-pon,wlen-pon],color='r',linestyle='--')
    plt.plot([7.5,7.5],[0,wlen],color='w',linestyle='--')
    if cxa.side==1:
        plt.plot([11.5,11.5],[0,wlen-pon],color='r',linestyle='--')
    else:
        plt.plot([3.5,3.5],[0,wlen-pon],color='r',linestyle='--')
        
    plt.figure()
    x = np.linspace(0,1,16)
    
    for iw,w in enumerate(wed_rand):
        plt.subplot(2,1,1)
        plt.plot(x+iw,w[:,0],color=[0,0,iw/len(wed_rand)])
        plt.plot(x+iw,w[:,1]+1.5,color=[0.8-(0.8*iw/len(wed_rand)),0.8-(0.8*iw/len(wed_rand)),0.8-(0.8*iw/len(wed_rand))])
        
        tp = phases[iw,0]
        tx = np.array([0,10*np.sin(tp)])
        ty = np.array([0,10*np.cos(tp)])
        plt.subplot(2,1,2)
        plt.plot(tx+iw+0.5,ty-1,color=[0,0,iw/len(wed_rand)])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.ylim([-10,10])
    plt.xlim([-10,len(wed_rand)+10])
    plt.subplot(2,1,1)
    plt.xlim([-10,len(wed_rand)+10])
    x = np.arange(0,len(wed_rand))
    plt.plot(x+0.5,np.mean(wed_rand[:,:,0],axis=1),color='b')
    plt.plot(x+0.5,np.mean(wed_rand[:,:,1],axis=1)+1.5,color='k')
    plt.plot([pon,pon],[0,2],color='r',linestyle='--')
#%% 

cxa = CX_a(datadirs[-3],regions=['eb','fsb_upper','fsb_lower'],denovo=False)
#%%
plt.close('all')
jumps = cxa.get_jumps()
for jump in jumps:
    wed_rand,pon,phases,traj = cxa.specific_jump_wedge(jump,fsb_names=['fsb_upper','eb'])
    
    wlen = wed_rand.shape[0]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(np.flipud(wed_rand[:,:,0]),interpolation=None,cmap='Blues')
    plt.plot([0.5,15.5],[wlen-pon,wlen-pon],color='r',linestyle='--')
    plt.plot([7.5,7.5],[0,wlen],color='w',linestyle='--')
    if cxa.side==1:
        plt.plot([11.5,11.5],[0,wlen-pon],color='r',linestyle='--')
    else:
        plt.plot([3.5,3.5],[0,wlen-pon],color='r',linestyle='--')
    plt.xticks([])
    plt.subplot(1,2,2)
    plt.imshow(np.flipud(wed_rand[:,:,1]),interpolation=None,cmap='Greys')
    plt.plot([0.5,15.5],[wlen-pon,wlen-pon],color='r',linestyle='--')
    plt.xticks([])
    plt.plot([0.5,15.5],[wlen-pon,wlen-pon],color='r',linestyle='--')
    plt.plot([7.5,7.5],[0,wlen],color='w',linestyle='--')
    if cxa.side==1:
        plt.plot([11.5,11.5],[0,wlen-pon],color='r',linestyle='--')
    else:
        plt.plot([3.5,3.5],[0,wlen-pon],color='r',linestyle='--')
        
    plt.figure()
    x = np.linspace(0,1,16)
    
    for iw,w in enumerate(wed_rand):
        plt.subplot(2,1,1)
        plt.plot(x+iw,w[:,0],color=[0,0,iw/len(wed_rand)])
        plt.plot(x+iw,w[:,1]+1.5,color=[0.8-(0.8*iw/len(wed_rand)),0.8-(0.8*iw/len(wed_rand)),0.8-(0.8*iw/len(wed_rand))])
        
        tp = phases[iw,0]
        tx = np.array([0,10*np.sin(tp)])
        ty = np.array([0,10*np.cos(tp)])
        plt.subplot(2,1,2)
        plt.plot(tx+iw+0.5,ty-20,color=[0,0,iw/len(wed_rand)])
        
        tp = phases[iw,1]
        tx = np.array([0,10*np.sin(tp)])
        ty = np.array([0,10*np.cos(tp)])
        plt.plot(tx+iw+0.5,ty,color=[0.8-(0.8*iw/len(wed_rand)),0.8-(0.8*iw/len(wed_rand)),0.8-(0.8*iw/len(wed_rand))])
        
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.ylim([-30,10])
    plt.xlim([-10,len(wed_rand)+10])
    plt.subplot(2,1,1)
    plt.xlim([-10,len(wed_rand)+10])
    x = np.arange(0,len(wed_rand))
    plt.plot(x+0.5,np.mean(wed_rand[:,:,0],axis=1),color='b')
    plt.plot(x+0.5,np.mean(wed_rand[:,:,1],axis=1)+1.5,color='k')
    plt.plot([pon,pon],[0,2],color='r',linestyle='--')
    
    plt.figure()
    xarray = np.array([-10,0,0,-10])*cxa.side*-1
    yarray = np.array([-50,-50,0,0])
    plt.fill(xarray,yarray,color=[0.7,0.7,0.7])
    plt.fill(xarray+3*cxa.side,-yarray,color=[0.7,0.7,0.7])
    plt.plot(traj[:,0],traj[:,1],color='k')
#%%  max -min wedge inside vs outside jumps
for datadir in datadirs:
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    jumps = cxa.get_jumps()
    r = 'fsb_upper'
    wedges = cxa.pdat['wedges_' +r]
    phase = cxa.pdat['offset_'+r+'_phase']*cxa.side*-1
    wmxmin = np.max(wedges,axis=1)-np.min(wedges,axis=1)
    
    indx = np.array([],dtype='int')
    outdx = np.array([],dtype='int')
    for j in jumps:
        indx = np.append(indx,np.arange(j[0],j[1]))
        outdx = np.append(outdx,np.arange(j[1],j[2]))
    inphase = phase[indx]
    inwmn = wmxmin[indx]
    
    ophase = phase[outdx]
    onwmn = wmxmin[outdx]
    bins = np.linspace(-np.pi,np.pi,9)
    plotmat = np.zeros((len(bins)-1,2))
    for i,b in enumerate(bins[:-1]):
        bdx = np.logical_and(inphase>b,inphase<bins[i+1])
        plotmat[i,0] = np.mean(inwmn[bdx])
        bdx = np.logical_and(ophase>b,ophase<bins[i+1])
        plotmat[i,1] = np.mean(onwmn[bdx])
    plt.plot(plotmat[:,0],color='r')
    plt.plot(plotmat[:,1],color='b')

#%%
plt.close('all')
jumps = cxa.get_jumps()

regions = ['fsb_upper','eb']

for i,datadir in enumerate(datadirs):
#datadir = datadirs[5]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    jumps = cxa.get_jumps()
    
    for ir,r in enumerate(regions):
        
        phase = cxa.pdat['phase_'+r]
        wedge = cxa.pdat['wedges_' +r]
        phasesmth = ug.savgol_circ(phase,30,3)
        pwrap = fn.unwrap(phase)
        psmwrap = fn.unwrap(phasesmth)
        dpdt = np.diff(psmwrap)/0.01
        for j in jumps:
            rdx = np.arange(j[0],j[2])
            wmn = np.mean(wedge[rdx,:],axis=1)
            wmnm = np.max(wedge[rdx,:],axis=1)-np.min(wedge[rdx,:],axis=1)
            pdt = dpdt[rdx]
            
            plt.figure(10+i)
            plt.subplot(1,2,ir+1)
            plt.scatter(wmn,wmnm,color='k',s=5,alpha=0.3)
            
            plt.figure(i)
            plt.subplot(1,2,ir+1)
            plt.scatter(np.abs(pdt),wmnm,color='k',s=5,alpha=0.3)
            
        plt.xscale('log')
        plt.title(r)
        plt.ylabel('Max-min df/F')
        plt.xlabel('dphase/dt')
#%%
plt.close('all')
angles = np.linspace(-np.pi,np.pi,16)
savedir = r'Y:\Presentations\2025\06_LabMeeting\FC2'
for datadir in datadirs:
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    weds = np.sum(cxa.pdat['wedges_fsb_upper']*np.sin(angles),axis=1)
    wedc = np.sum(cxa.pdat['wedges_fsb_upper']*np.cos(angles),axis=1)
    pva  = np.sqrt(weds**2+wedc**2)
    pvaz = (pva-np.mean(pva))/np.std(pva)
    fci = fci_regmodel(pvaz,cxa.ft2,cxa.pv2)
    plt.figure()
    fci.mean_traj_nF_jump(pvaz,plotjumps=True,colormap='coolwarm',cmx=1)
    plt.savefig(os.path.join(savedir,'AllJumpsPVAz'+cxa.name+'.pdf'))
    plt.figure()
    fci.mean_traj_heat_jump(pvaz,set_cmx=True,cmx=1)
    plt.xlim([-50,50])
    plt.ylim([-50,50])
    plt.savefig(os.path.join(savedir,'MeanJumpsPVAz'+cxa.name+'.pdf'))
    
    wmean=  np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
    wz = (wmean-np.mean(wmean))/np.std(wmean)
    plt.figure()
    fci.mean_traj_nF_jump(wz,plotjumps=True,colormap='coolwarm',cmx=1)
    plt.savefig(os.path.join(savedir,'AllJumpsWedgeMeanz'+cxa.name+'.pdf'))
    plt.figure()
    fci.mean_traj_heat_jump(wz,set_cmx=True,cmx=1)
    plt.xlim([-50,50])
    plt.ylim([-50,50])
    plt.savefig(os.path.join(savedir,'MeanJumpsWedgeMeanz'+cxa.name+'.pdf'))
#%% Heading vs FC2 phase
plt.close('all')
angles = np.linspace(-np.pi,np.pi,18)
savedir = r'Y:\Presentations\2025\06_LabMeeting\FC2'
plt.figure()
offset = 0
ampall = np.zeros((len(angles)-1,len(datadirs)))
for i,datadir in enumerate(datadirs):
    
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    jumps = cxa.get_jumps()
    epg = cxa.pdat['offset_eb_phase'].to_numpy()*cxa.side*-1
    FC2 = cxa.pdat['offset_fsb_upper_phase'].to_numpy()*cxa.side*-1
    
    indx = np.array([],dtype='int')
    outdx = np.array([],dtype='int')
    for j in jumps:
        indx = np.append(indx,np.arange(j[0],j[1]))
        outdx = np.append(outdx,np.arange(j[1],j[2]))
    out_epg = epg[outdx]
    out_fc2 = FC2[outdx]
    tamp = cxa.pdat['amp_fsb_upper'][outdx]
    for ai,a in enumerate(angles[:-1]):
        adx = np.logical_and(out_epg>a,out_epg<angles[ai+1])
        fcmn = stats.circmean(out_fc2[adx],high=np.pi,low=-np.pi)
        mamp = np.mean(tamp[adx])
        #print(a,angles[ai+1],fcmn)
        #print(out_fc2[adx][0])
        x = np.array([0,np.sin(fcmn)*mamp])+offset
        y = np.array([0,np.cos(fcmn)*mamp])
        plt.figure(2)
        plt.scatter((a+angles[ai+1])/2,mamp,color=[1-ai/17,0,ai/17])
        plt.figure(1)
        plt.plot(x,y,color=[1-ai/17,0,ai/17])
        ampall[ai,i] = mamp
    offset = offset+0.18
    g = plt.gca()
    g.set_aspect('equal')
 
angles = np.linspace(-np.pi,np.pi,17)
for ai,a in enumerate(angles):
    x = np.array([0,np.sin(a)*0.1])+offset
    y = np.array([0,np.cos(a)*0.1])
    plt.plot(x,y,color=[1-ai/16,0,ai/16])

plt.savefig(os.path.join(savedir,'PowerAndFC2Phase_withEPG.png'))
plt.savefig(os.path.join(savedir,'PowerAndFC2Phase_withEPG.pdf'))
plt.figure(2)
plt.plot(angles,np.nanmean(ampall,axis=1),color='k')
plt.xlabel('EPG phase (deg)')
plt.ylabel('FC2 amplitude (pva)')
plt.savefig(os.path.join(savedir,'Phase_power.png'))
plt.savefig(os.path.join(savedir,'Phase_power.pdf'))
#%%    

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
wedmxmn = np.max(wednorm,axis=1)-np.min(wednorm,axis=1)
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
fci.example_trajectory_jump(pvan_z,cxa.ft,cmin=0,cmax=6)
fci.mean_traj_nF_jump(wedmxmn,plotjumps=True,colormap='coolwarm')
fci.mean_traj_heat_jump(wedmxmn)
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
