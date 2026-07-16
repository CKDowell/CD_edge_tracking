# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:02:57 2026

@author: dowel
"""

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from analysis_funs.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
from src.utilities import funcs as fn
from scipy import stats
from Utilities.utils_general import utils_general as ug
from Utilities.utils_plotting import uplt
plt.rcParams['font.sans-serif'] = 'Arial'
from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper

groups = {
    "FC2": ["Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2",
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial5",
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial2",
    r"Y:\Data\FCI\Hedwig\FC2_maimon2\250128\f1\Trial4"
    ],
    
    "hDeltaC_FC2_EPG": [
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f1\Trial3', # 6 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260426\f1\Trial1', # 8 jumps
        #r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260425\f2\Trial3', # 3 jumps
        #r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260401\f1\Trial2', # Some nice tracking 3 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260429\f2\Trial1',# 17 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f1\Trial2', # 6 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260623\f1\Trial2', # 6 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260707\f1\Trial3' # 7 jumps
        ],
    

    
}
all_flies = {}

for group_name, datadirs in groups.items():
    all_flies[group_name] = {}
    
    for i, datadir in enumerate(datadirs):
        print(f"[{group_name}] {datadir}")
        if group_name=='hDeltaC_FC2':
            etp = ET_paper(datadir,regions=['fsb1_ch1','fsb2_ch2'])
        elif group_name=='hDeltaC_FC2_EPG':
            etp = ET_paper(datadir,regions=['eb_ch1','fsb1_ch1','fsb2_ch2'])
        elif group_name=='PFN_pm':
            etp = ET_paper(datadir,regions= ['pfn_l_16','pfn_r_16','pfn_l','pfn_r'] )
        else:
            etp = ET_paper(datadir)
        all_flies[group_name][i] = etp
#%% PVA vs goal offset
dchoice = ['FC2','hDeltaC_FC2_EPG']
fcname = ['fsb_upper','fsb1_ch1']
ebname = ['eb','eb_ch1']
bins = np.linspace(0,.25,26)
plotbins = bins[1:]-np.mean(np.diff(bins))/2
plt.close('all')
amp_choice = ['mean','PVA','co_norm']
index = 'all'
# All data
for d1,d in enumerate(dchoice):
    plt.figure()
    data = all_flies[d]
    
    for ai, a in enumerate(amp_choice):
        
        plt.subplot(2,2,ai+1)
        for i in data:
            etp = data[i]
            fc2 = etp.cxa.pdat['phase_'+fcname[d1]]
            epg = etp.cxa.pdat['phase_'+ebname[d1]]
            offset = 180*np.abs(ug.circ_subtract(fc2,epg))/np.pi
            
            if a =='PVA':
                pva = ug.get_pvas(etp.cxa.pdat['wedges_'+fcname[d1]])
                bins = np.linspace(0,.25,26)
                plotbins = bins[1:]-np.mean(np.diff(bins))/2
            elif a =='mean':
                pva = np.mean(etp.cxa.pdat['wedges_'+fcname[d1]],axis=1)
                bins = np.linspace(0,1,21)
                plotbins = bins[1:]-np.mean(np.diff(bins))/2
            elif a =='co_norm':
                pva =  ug.get_pvas(etp.cxa.pdat['wedges_'+fcname[d1]]) / np.mean(etp.cxa.pdat['wedges_'+fcname[d1]],axis=1)
                bins = np.linspace(0,1,21)
                plotbins = bins[1:]-np.mean(np.diff(bins))/2
                
            if index=='all':
                dx = np.ones(len(fc2),dtype=int)==1
            elif dx=='instrip':
                dx = etp.cxa.ft2['instrip'].to_numpy()==1
                # option to roll to account for delay
                #dx = np.roll(dx,5)
                #dx[:5] =0
            elif dx== 'outstrip':
                dx = etp.cxa.ft2['instrip'].to_numpy()<1
                
            fc2 = fc2[dx]
            epg = epg[dx]
            offset = offset[dx]
            pva = pva[dx]    
            
                
            if i==0:
                all_data = np.zeros((len(plotbins),len(data)))
            bin_idx = np.digitize(pva, bins)
            tdat = np.array([
                offset[bin_idx==i].mean() if np.any(bin_idx==i) else np.nan 
                for i in range(len(bins)-1)
                ])
            
            plt.plot(plotbins,tdat,color='k',alpha=.5)
            all_data[:,i] = tdat
        dmean = np.nanmean(all_data,axis=1)
        plt.plot(plotbins,dmean,color='k',linewidth=2)
        plt.xlabel(a)
        plt.ylabel('FC2 - EPG offset')
        plt.ylim([0,150])
        plt.yticks(np.arange(0,160,20))
#%% hDeltaC and FC2 offset
# Similar PVA and mean analysis for offset between these signals
dchoice = ['hDeltaC_FC2_EPG']
fcname = ['fsb1_ch1']
ebname = ['fsb2_ch2']
bins = np.linspace(0,.25,26)
plotbins = bins[1:]-np.mean(np.diff(bins))/2
plt.close('all')
amp_choice = ['mean','PVA','co_norm']
index = 'outstrip'
# All data
for d1,d in enumerate(dchoice):
    plt.figure()
    data = all_flies[d]
    
    for ai, a in enumerate(amp_choice):
        
        plt.subplot(2,2,ai+1)
        for i in data:
            etp = data[i]
            fc2 = etp.cxa.pdat['phase_'+fcname[d1]]
            hdc = etp.cxa.pdat['phase_'+ebname[d1]]
            offset = 180*np.abs(ug.circ_subtract(fc2,hdc))/np.pi
            
            if a =='PVA':
                pva = ug.get_pvas(etp.cxa.pdat['wedges_'+ebname[d1]])
                bins = np.linspace(0,.25,26)
                plotbins = bins[1:]-np.mean(np.diff(bins))/2
            elif a =='mean':
                pva = np.mean(etp.cxa.pdat['wedges_'+ebname[d1]],axis=1)
                bins = np.linspace(0,1,21)
                plotbins = bins[1:]-np.mean(np.diff(bins))/2
            elif a =='co_norm':
                pva =  ug.get_pvas(etp.cxa.pdat['wedges_'+ebname[d1]]) / np.mean(etp.cxa.pdat['wedges_'+ebname[d1]],axis=1)
                bins = np.linspace(0,1,21)
                plotbins = bins[1:]-np.mean(np.diff(bins))/2
                
            if index=='all':
                dx = np.ones(len(fc2),dtype=int)==1
            elif index=='instrip':
                dx = etp.cxa.ft2['instrip'].to_numpy()==1
                # option to roll to account for delay
                #dx = np.roll(dx,5)
                #dx[:5] =0
            elif index== 'outstrip':
                dx = etp.cxa.ft2['instrip'].to_numpy()<1
                # option to roll to account for delay
                dx = np.roll(dx,5)
                dx[:5] =0
                
            fc2 = fc2[dx]
            hdc = hdc[dx]
            offset = offset[dx]
            pva = pva[dx]    
            
                
            if i==0:
                all_data = np.zeros((len(plotbins),len(data)))
            bin_idx = np.digitize(pva, bins)
            tdat = np.array([
                offset[bin_idx==i].mean() if np.any(bin_idx==i) else np.nan 
                for i in range(len(bins)-1)
                ])
            
            plt.plot(plotbins,tdat,color='k',alpha=.5)
            all_data[:,i] = tdat
        dmean = np.nanmean(all_data,axis=1)
        plt.plot(plotbins,dmean,color='k',linewidth=2)
        plt.xlabel(a)
        plt.ylabel('FC2 - EPG offset')
       # plt.ylim([0,150])
        plt.yticks(np.arange(0,160,20))


#%% Allocentric phase and PVA
plt.close('all')
dchoice = ['FC2','hDeltaC_FC2_EPG']
fcname = ['fsb_upper','fsb1_ch1']
ringbins = np.linspace(-np.pi,np.pi,101)
bins = np.linspace(-np.pi,np.pi,17)
pbins = bins[1:] - np.mean(np.diff(bins))/2
pbins = np.append(pbins,pbins[0])
xn = np.sin(pbins)
yn = np.cos(pbins)
xr = np.sin(ringbins)
yr = np.cos(ringbins)
colours = uplt.columnar_colours()
index = 'jumps'
for d1,d in enumerate(dchoice):
    data = all_flies[d]
    plt.figure()
    allmean = np.zeros((len(pbins),len(data)))
    for i in data:
        etp =data[i]
        jumps = etp.cxa.get_jumps()
        
            
        fc2 = etp.cxa.pdat['offset_'+fcname[d1]+'_phase'].to_numpy()*etp.cxa.side*-1
        pva = ug.get_pvas(etp.cxa.pdat['wedges_'+fcname[d1]])
        if index=='all':
            dx = np.ones(len(fc2),dtype=int)==1
        elif index=='jumps':
            dx = np.concatenate([np.arange(j[1],j[2],dtype='int') for j in jumps])
        
        fc2 = fc2[dx]
        pva = pva[dx]
        bin_idx = np.digitize(fc2,bins)-1
        tdat = np.array([
            pva[bin_idx==i].mean() if np.any(bin_idx==i) else np.nan 
            for i in range(len(bins)-1)
            ])
        tdat = np.append(tdat,tdat[0])
        
        x = np.sin(fc2)*pva
        y = np.cos(fc2)*pva
        #plt.figure()
        #plt.scatter(x,y,s=1,color='k',alpha=.2)
        xt = xn*tdat
        yt = yn*tdat
        plt.plot(xt,yt,color=colours[2,:],alpha=.5)
        allmean[:,i] = tdat
    plt.gca().set_aspect('equal')
    mn = np.mean(allmean,axis=1)
    plt.plot(xr*.1,yr*.1,color='k',linestyle='--',alpha=.5)
    plt.plot(xr*.075,yr*.075,color='k',linestyle='--',alpha=.5)
    plt.plot(xr*.05,yr*.05,color='k',linestyle='--',alpha=.5)
    plt.plot(xr*.025,yr*.025,color='k',linestyle='--',alpha=.5)
    plt.plot([-.125,.125],[0,0],color='k')
    plt.plot([0,0],[-.125,.125],color='k')
    plt.plot(xn*mn,yn*mn,color=colours[2,:],linewidth=3)
    plt.fill(xr[:51]*.1,yr[:51]*.1,color='r',zorder=-1,alpha=.25)
    plt.title(d)
    plt.xlim([-.17,.17])
    plt.ylim([-.17,.17])
#%%     
dchoice = 'FC2'
fly =5
# 4 has very nice phase transitions
etp = all_flies[dchoice][fly]
region = 'fsb_upper'
#region = 'fsb1_ch1'
#ebregion = 'eb_ch1'
ebregion = 'eb'
pva = ug.get_pvas(etp.cxa.pdat['wedges_'+region])
etp.cxa.plot_traj_arrow_heat([region],pva,a_sep=4,colormap='coolwarm',cmin=0,cmax=.2)


#%% Phase and PVA simple plots
cxa = etp.cxa
plt.figure()
colours = uplt.columnar_colours()

u = ug()
x = cxa.pv2['relative_time'].to_numpy()
_,_,dd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),x)

ins = etp.cxa.ft2['instrip'].to_numpy()

fc2 = etp.cxa.pdat['phase_'+region]
fc2_n = etp.cxa.pdat['offset_'+region+'_phase'].to_numpy()

#hdc_n = etp.cxa.pdat['offset_'+'fsb2_ch2'+'_phase'].to_numpy()
#hdc = etp.cxa.pdat['phase_'+'fsb2_ch2']
eb = etp.cxa.pdat['phase_'+ebregion]
heading = etp.cxa.ft2['ft_heading'].to_numpy()
amp = np.mean(etp.cxa.pdat['wedges_'+region],axis=1)
dfc2 = np.diff(fc2)
deb = np.diff(eb)
dratio = dfc2


offset = 1*ug.circ_subtract(fc2,eb)/np.pi
plt.plot(x,pva*5,color='k')
#plt.plot(x,pva2*5,color='b')
plt.plot(x,ins,color='r')
plt.plot(x[1:],-.5+dd/20,color=[.5,.5,.5])
plt.scatter(x,-1.5+eb/np.pi,color='k',s=3,zorder=6)
plt.scatter(x,-1.5+fc2/np.pi,color=colours[2,:],s=3,zorder=6)
#plt.scatter(x,-1.5+hdc/np.pi,color=colours[1,:],s=3,zorder=6)
plt.scatter(x,np.abs(offset),color=colours[2,:],s=5,zorder=5)


#%% Compare RDP trajectory error and PVA
# Note: significant correlation for returns to plume
plt.close('all')
dchoice = ['FC2','hDeltaC_FC2_EPG']
f = 0

for di,d in enumerate(dchoice):
    t_flies = all_flies[d]
    if d=='FC2':
        regions = ['eb','fsb_upper']
    else:
        regions = ['eb_ch1','fsb1_ch1']
    for fly in t_flies:
        f = f+1
        # 4 has very nice phase transitions
        etp = all_flies[d][fly]
        data = etp.turn2plume_metrics(epsilon=2,regions=regions)
        tphase = data['phase']
        if len(tphase)==0:
            continue
        tpva = data['pva']
        rdpang = data['rdpang']
        tamp = data['mean_amp']
        x = np.zeros((len(tphase),2))
        x[:,0] = tphase[:,1]
        x[:,1] = rdpang
        y = np.ones((len(tphase),2))*tpva[:,1][:,np.newaxis]
        # plt.figure()
        # plt.plot(x.T,y.T,color='k')
        # plt.plot(tphase.T,y.T,color='k')
        # plt.scatter(tphase[:,1],tpva[:,1],s=30,color='b')
        # plt.scatter(tphase[:,0],tpva[:,1],s=30,color='k')
        # plt.scatter(rdpang,tpva[:,1],s=30,color='r')
        # plt.xlim([-np.pi,np.pi])
        
        if fly==0 and di ==0:
            tdata  = ug.circ_subtract(tphase[:,1],rdpang)
            tdata2 = tpva[:,1]
            tdata3 = tamp[:,1]
            fall = np.zeros(len(tamp))+f
        else:
            tdata = np.append(tdata,ug.circ_subtract(tphase[:,1],rdpang))
            tdata2 = np.append(tdata2,tpva[:,1])
            tdata3 = np.append(tdata3,tamp[:,1])
            fall = np.append(fall,np.zeros(len(tamp))+f)

plt.figure()
plt.scatter(tdata,tdata2,c = fall,cmap = 'Paired')
mns,edges,num = stats.binned_statistic(tdata,tdata2,bins=np.linspace(-np.pi,np.pi,11))
ex = edges[1:]-np.mean(np.diff(edges))/2
plt.plot(ex,mns,color='k')
plt.figure()
plt.scatter(np.abs(tdata),tdata2,color='k')
pr = stats.pearsonr(np.abs(tdata),tdata2)
import sklearn.linear_model as lm
reg = lm.LinearRegression(fit_intercept=True)
reg.fit(np.abs(tdata)[:,np.newaxis],tdata2)
yfit = reg.predict(np.array([0,np.pi]).reshape(-1,1))
plt.plot([0,np.pi],yfit,color='r')
plt.ylabel('PVA (z scored)')
plt.xlabel('trajectory - FC2 error (radians)')
plt.text(np.pi/2,.2,"Peason's r: " + str(ug.round_to_sig_figs(np.array([pr[0]]), 2)[0]),color='r')
plt.text(np.pi/2,.175,"p : " + str(ug.round_to_sig_figs(np.array([pr[1]]), 2)[0]),color='r')
plt.title('To plume')
#plt.ylim([-3.25,3.25])

plt.figure()
plt.scatter(np.abs(tdata),tdata3,color='k')

for di,d in enumerate(dchoice):
    t_flies = all_flies[d]
    if d=='FC2':
        regions = ['eb','fsb_upper']
    else:
        regions = ['eb_ch1','fsb1_ch1']
    for fly in t_flies:
        
        # 4 has very nice phase transitions
        etp = all_flies[d][fly]
        data = etp.turn2plume_metrics(epsilon=2,regions=regions,to_plume=False)
        tphase = data['phase']
        if len(tphase)==0:
            continue
        tpva = data['pva']
        rdpang = data['rdpang']
        x = np.zeros((len(tphase),2))
        x[:,0] = tphase[:,1]
        x[:,1] = rdpang
        y = np.ones((len(tphase),2))*tpva[:,1][:,np.newaxis]
        # plt.figure()
        # plt.plot(x.T,y.T,color='k')
        # plt.plot(tphase.T,y.T,color='k')
        # plt.scatter(tphase[:,1],tpva[:,1],s=30,color='b')
        # plt.scatter(tphase[:,0],tpva[:,1],s=30,color='k')
        # plt.scatter(rdpang,tpva[:,1],s=30,color='r')
        # plt.xlim([-np.pi,np.pi])
        
        if fly==0 and di ==0:
            tdata  = ug.circ_subtract(tphase[:,1],rdpang)
            tdata2 = tpva[:,1]
        else:
            tdata = np.append(tdata,ug.circ_subtract(tphase[:,1],rdpang))
            tdata2 = np.append(tdata2,tpva[:,1])


plt.figure()
plt.scatter(tdata,tdata2,color='k')
plt.figure()
plt.scatter(np.abs(tdata),tdata2,color='k')
pr = stats.pearsonr(np.abs(tdata),tdata2)
import sklearn.linear_model as lm
reg = lm.LinearRegression(fit_intercept=True)
reg.fit(np.abs(tdata)[:,np.newaxis],tdata2)
yfit = reg.predict(np.array([0,np.pi]).reshape(-1,1))
plt.plot([0,np.pi],yfit,color='r')
plt.ylabel('PVA (z scored)')
plt.xlabel('trajectory - FC2 error (radians)')
plt.text(np.pi/2,.15,"Peason's r: " + str(ug.round_to_sig_figs(np.array([pr[0]]), 2)[0]),color='r')
plt.text(np.pi/2,.125,"p : " + str(ug.round_to_sig_figs(np.array([pr[1]]), 2)[0]),color='r')
#plt.ylim([-3.25,3.25])
plt.title('Away from plume')

#%% Identify transitions of FC2 phase to hDeltaC phase
plt.close('all')
etp = all_flies['hDeltaC_FC2_EPG'][4]
ins = etp.cxa.ft2['instrip'].to_numpy()
hdc = etp.cxa.pdat['offset_fsb2_ch2_phase'].squeeze()
fc2 = etp.cxa.pdat['offset_fsb1_ch1_phase'].squeeze()
epgw = etp.cxa.pdat['wedges_eb_ch1']
hdcw = etp.cxa.pdat['wedges_fsb2_ch2']
amphdc = np.mean(etp.cxa.pdat['wedges_fsb2_ch2'],axis=1)
pvahdc = ug.get_pvas(etp.cxa.pdat['wedges_fsb2_ch2'])
fc2w = etp.cxa.pdat['wedges_fsb1_ch1']
t = np.arange(0,len(hdc))/10
cv = ug.rowwise_cov(fc2w,hdcw)
cve  = ug.rowwise_cov(fc2w,epgw)
plt.plot(t,cv*50,color='k')
#plt.plot(t,cve*50,color=[.5,.5,.5])
plt.plot(t,t*0,color='r')
colours = uplt.columnar_colours()
plt.scatter(t,fc2,color=colours[2,:],s=2)
plt.plot(t,ins,color='r')
plt.scatter(t,hdc,color=colours[1,:],s=2)
plt.plot(t,amphdc-np.pi-1.5
         ,color=colours[1,:])
plt.plot(t,pvahdc*5-np.pi-3
         ,color=colours[1,:])
u = ug()
_,_,dd = u.get_velocity(etp.cxa.ft2['ft_posx'].to_numpy(),etp.cxa.ft2['ft_posy'].to_numpy(),t)
plt.plot(t[1:],dd/10-np.pi-2,color=[.5,.5,.5])
etp.cxa.plot_traj_arrow_heat(['fsb1_ch1','fsb2_ch2'],cv,a_sep=4,colormap='coolwarm',cmin=-.015,cmax=.015)
#%%
import sklearn.linear_model as lm
reg = lm.LinearRegression(fit_intercept=True)
y = cv.copy()
dd2= np.append(0,dd)
X  = np.append(pvahdc[:,np.newaxis],amphdc[:,np.newaxis],axis=1)
X = np.append(X,dd2[:,np.newaxis],axis=1)

X = X/np.std(X,axis=0)

#y[y<0] = 0
reg.fit(X[ins==0,:],y[ins==0])
r2 = reg.score(X[ins==0,:],y[ins==0])
yp = reg.predict(X)
plt.figure()
plt.plot(y)
plt.plot(yp)

pmat = np.append(X,y[:,np.newaxis],axis=1)
C = np.corrcoef(pmat[ins==0,:].T)
plt.figure()
plt.scatter(cv,pvahdc,s=1)

#%% Graveyard
#%% Identify 7 shaped trajectories
# Fit three points and interpolate between
# Get MSE and set threshold
# Use 7 point for timepoint of decision making
plt.close('all')
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
x,y = ug.fictrac_repair(x, y)
xy = np.append(x[:,np.newaxis],y[:,np.newaxis],axis=1)
jumps = cxa.get_jumps()
_,_,dd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),x)


still = ug.get_still(1,np.abs(dd),merg_thresh=3,minsize=2)

for i,j in enumerate(jumps):
    dx = np.arange(j[1],j[2])
    
    t = dx-dx[0]
    #txyo = xy[dx,:]
    txy = xy[dx,:]
    tp = fc2_n[dx]*-cxa.side
    th = heading[dx]*-cxa.side
    tpva = pva[dx]
    tstill = still[dx]
    
    
    txy = txy-txy[0,:]
    txy[:,0] = txy[:,0]*-cxa.side
    
    st = txy[0,:]
    ed = txy[-1,:]
    plt.figure()
    plt.subplot(2,2,1)
    
    msqes = np.zeros(len(txy)-2)
    
    
    mx = np.argmax(txy[:,0])
    plt.plot(txy[:,0],txy[:,1],color='k')
    for di,d in enumerate(dx[1:-1]):
        # simple 7 fit
        m = txy[di+1]
        
        x1 = np.interp(t[:di+1],np.array([t[0],t[di]]),np.array([st[0],m[0]]))
        y1 = np.interp(t[:di+1],np.array([t[0],t[di]]),np.array([st[1],m[1]]))
        x2 = np.interp(t[di+1:],np.array([t[di+1],t[-1]]),np.array([m[0],ed[0]]))
        y2 = np.interp(t[di+1:],np.array([t[di+1],t[-1]]),np.array([m[1],ed[1]]))
        px = np.append(x1,x2)
        py = np.append(y1,y2)
        pxy = np.append(px[:,np.newaxis],py[:,np.newaxis],axis=1)
        sqe = (txy-pxy)**2
        msqes[di] = np.mean(sqe[:])
        plt.plot(pxy[:,0],pxy[:,1],color='b',alpha=.25)
        
        
        
        
    mn = np.argmin(msqes)+1
    
    di = mn
    m = txy[di]
    x1 = np.interp(t[:di+1],np.array([t[0],t[di]]),np.array([st[0],m[0]]))
    y1 = np.interp(t[:di+1],np.array([t[0],t[di]]),np.array([st[1],m[1]]))
    
    x2 = np.interp(t[di+1:],np.array([t[di+1],t[-1]]),np.array([m[0],ed[0]]))
    y2 = np.interp(t[di+1:],np.array([t[di+1],t[-1]]),np.array([m[1],ed[1]]))
    
    px = np.append(x1,x2)
    py = np.append(y1,y2)
    pxy = np.append(px[:,np.newaxis],py[:,np.newaxis],axis=1)
    
    mdx = np.arange(np.max(np.array([mn+1-15,0])),np.min(np.array([mn+1+15,len(dx)])))
    
    xymini = txy[mdx,:]
    m2 = mdx[np.argmax(xymini[:,0])]
    
    plt.plot(pxy[:,0],pxy[:,1],color='r')
    plt.scatter(txy[m2,0],txy[m2,1],color='g')
    plt.gca().set_aspect('equal')
    plt.title(msqes[mn-1])
    plt.subplot(2,2,2)
    plt.plot(msqes)
    plt.scatter(mx-1,msqes[mx-1])
    plt.scatter(m2,msqes[m2],color='r')
    
    
    plt.subplot(2,2,3)
    plt.plot(th,color='k')
    plt.plot(tp,color='b')
    plt.plot([m2,m2],[-np.pi,np.pi],color='r')
    plt.plot(tstill,color='k')
    plt.subplot(2,2,4)
    plt.plot(tpva,color='k')
    plt.scatter(m2,tpva[m2],color='r')
    plt.ylim([0,.2])
#%%  Regression fit for 7
#regression 7 fit     

plt.close('all')
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
x,y = ug.fictrac_repair(x, y)
xy = np.append(x[:,np.newaxis],y[:,np.newaxis],axis=1)
jumps = cxa.get_jumps()
import sklearn.linear_model as lm

reg = lm.LinearRegression(fit_intercept=True)
for i,j in enumerate(jumps):
    dx = np.arange(j[1],j[2])
    
    t = dx-dx[0]
    #txyo = xy[dx,:]
    txy = xy[dx,:]
    st = txy[0,:]
    ed = txy[-1,:]
    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.plot(txy[:,0],txy[:,1],color='k')
    r2s = np.zeros((len(dx[5:-5]),2))
    txy = txy-txy[0,:]
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(txy[:,0],txy[:,1],color='k')
    for di,d in enumerate(dx[5:-5]):
        
   
        reg.fit(txy[:di+6,0].reshape(-1,1),txy[:di+6,1])
        r2s[di,0] = reg.score(txy[:di+6,0].reshape(-1,1),txy[:di+6,1])
        
        reg.fit(txy[di+5:,0].reshape(-1,1),txy[di+5:,1])
        r2s[di,1] = reg.score(txy[di+5:,0].reshape(-1,1),txy[di+5:,1])
    
    r2m = np.mean(r2s,axis=1)
    mn = np.argmax(r2m)+5
    plt.scatter(txy[mn,0],txy[mn,1],color='r')        
    plt.subplot(1,2,2)
    plt.plot(r2s)
    plt.plot(r2m,color='k')


#%% In service of turning point estimation 

# Note, below is a little insufficient

# 1. smooth trajectory
# 2. get heading
# 3. get angular velocity
plt.close('all')
cxa = etp.cxa
ins = cxa.ft2['instrip'].to_numpy()
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
x,y = ug.fictrac_repair(x, y)
plt.plot(x,y,color='k')
plt.scatter(x[ins>0],y[ins>0],color=[.5,.5,.5])

# Savgol filter
xs = sg.savgol_filter(x, 20, 4)
ys = sg.savgol_filter(y, 20, 4)
plt.plot(xs,ys,color='b')
xs = sg.savgol_filter(x, 10, 5)
ys = sg.savgol_filter(y, 30, 5)
plt.plot(xs,ys,color='g')
plt.gca().set_aspect('equal')

# Kalman filter
xy = np.append(x[:,np.newaxis],y[:,np.newaxis],axis=1)

from filterpy.kalman import KalmanFilter
f = KalmanFilter(dim_x = 2,dim_z = 2) # two measurements, 2 states
f.x = np.array([[x[0],y[0]]]).T
f.H = np.eye(2) # Measurement function
f.F = np.eye(2) # State transition  assume states don't change
f.P = np.cov(x,y)*.5 # Covariance
f.R = np.eye(2)*5# Measurement noise
from filterpy.common import Q_discrete_white_noise
#f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)


xyk = xy*0 
for i,txy in enumerate(xy):
    f.predict()
    f.update(np.array([xy[i,:].T]))
    xyk[i,:] = f.x_post.squeeze()
xyk = np.append(xyk[1:,:],[xyk[-1,:]],axis=0) # roll the filter to account for one timestep delay
    
#plt.figure()
#plt.plot(x,y,color='k')
plt.plot(xyk[:,0],xyk[:,1],color='r')
#plt.gca().set_aspect('equal')
plt.figure()
plt.plot(x,color='k')
plt.plot(xyk[:,0],color='r')

# get heading
xyd = np.diff(xyk,axis=0)
xyd[np.abs(xyd)<.01] =0
kheading = np.angle(xyd[:,1]+xyd[:,0]*1j)
kheading = np.append(kheading,kheading[-1]) # again append extra value
plt.figure()
heading = cxa.ft2['ft_heading'].to_numpy()
plt.plot(kheading,color='r')
plt.plot(heading,color='k')

#%% Identify RDP turning points and plot on trajectory and timeserie to establish turn events
from rdp import rdp
heading = cxa.ft2['ft_heading'].to_numpy()
xy = np.append(cxa.ft2['ft_posx'].to_numpy()[:,np.newaxis],cxa.ft2['ft_posy'].to_numpy()[:,np.newaxis],axis=1)
av =ug.get_ang_velocity(cxa.ft2['ft_heading'].to_numpy(),cxa.pv2['relative_time'].to_numpy())
rxy = rdp(xyk,1.25,return_mask=True) # two slightly oversegments, but is good as afirs pass
rxyw = np.where(rxy)[0]
plt.figure()
plt.plot(xy[:,0],xy[:,1],color='k')
plt.scatter(xy[ins>0,0],xy[ins>0,1],color='r')
rdpxy = xy[rxy,:]
plt.plot(rdpxy[:,0],rdpxy[:,1],color=[0.5,0.5,.5])
for i in range(len(rdpxy)):
    plt.text(rdpxy[i,0],rdpxy[i,1],str(i))


plt.gca().set_aspect('equal')

angchange = np.zeros(len(rxyw)-2)
angchangedx = rxyw[1:-1]
angchangexy = rdpxy[1:-1,:]
tang = np.zeros(len(rxyw)-1)
for i,w in enumerate(rxyw[1:-1]):
    v1 = rdpxy[i:i+2,:]
    v2 = rdpxy[i+1:i+3,:]
    v1 = v1-v1[0,:]
    v2 = v2-v2[0,:]
    a1 = np.angle(v1[1,1]+v1[1,0]*1j)
    a2 = np.angle(v2[1,1]+v2[1,0]*1j)
    df = ug.circ_subtract(a2,a1)
    angchange[i] = df
    tang[i] =a1
    tang[i+1] =a2
plt.scatter(angchangexy[angchange<0,0],angchangexy[angchange<0,1],color='b')
    

plt.figure()
#plt.plot(x[1:],dd/10,color='k')
plt.plot(x,heading/np.pi,color='k')
rxyp = rxy.astype('float')
rxyp[angchangedx[angchange<0]] = -1
plt.plot(x,rxyp,color='r')

plt.plot(x,pva,color='m')
plt.plot(x,ins-2.5,color='r')
ymini = np.append(tang[:,np.newaxis],tang[:,np.newaxis],axis=1)/np.pi
xmini = np.append(x[rxyw[:-1],np.newaxis],x[rxyw[1:],np.newaxis],axis=1)
plt.plot(xmini.T,ymini.T,color='g')
plt.plot(x[1:],av/10)
for i,w in enumerate(rxyw):
    plt.text(x[w],1,str(i))
    
plt.figure()
plt.plot(x,heading,color='k')
plt.scatter(x,fc2_n,color='b',s=4,zorder=6)
plt.scatter(x,hdc_n,color='m',s=4,zorder=6)
plt.plot(xmini.T,ymini.T,color='g')
plt.plot(x,ins-4,color='r')
plt.plot(x,pva*5-4,color='k')
plt.plot(x,amp*4-5,color=[.5,.5,.5])
plt.plot(x,pva*amp*8-4,color=[.8,.5,.5])
#%% Get a phase velocity estimate that is reliable
plt.close('all')
plt.figure()
plt.scatter(x,fc2,color='k',s=3)
fc2s = ug.savgol_circ(fc2, 10, 3)
fc2s2 = ug.savgol_circ(fc2, 50, 3)
plt.plot(x,fc2s,color='k')
plt.plot(x,fc2s2,color='b')
ebw = np.unwrap(fc2s)
deb = np.diff(ebw)

plt.scatter(x[1:],deb,color='r',s=3,zorder=10)
plt.plot(x,x*0,color=[.5,.5,.5])
#prctile = stats.percentileofscore(np.abs(deb),np.abs(deb))/100
prctile = (stats.percentileofscore(deb,deb)-50)/100

cnum = 5
pconv = prctile[cnum:]
for i in range(cnum-1):
    pconv = pconv*prctile[cnum-1-i:-i-1]


#plt.plot(x[cnum+1:],pconv*100)

plt.plot(x[1:],-.5+dd/20,color=[.5,.5,.5])
rxy = rdp(xy,10,return_mask=True)


#%%
regions2 = ['eb_ch1','fsb1_ch1','fsb2_ch2']
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=True)
#%% Turn detection - use RDP to identify shifts in direction on trajectory then use these to assess phase transitions.
heading = cxa.ft2['ft_heading'].to_numpy()
hw = np.unwrap(heading)

grad = (np.mean(hw[-10:])-np.mean(hw[:10]))/len(hw)
sub = np.arange(0,len(hw))*grad
plt.plot(hw-sub)
step  = np.ones(20)
step[:10] =-1
tconv = np.convolve(hw,step)
plt.plot(tconv)

xy = np.append(cxa.ft2['ft_posx'].to_numpy()[:,np.newaxis],cxa.ft2['ft_posy'].to_numpy()[:,np.newaxis],axis=1)
from rdp import rdp
rxy = rdp(xy,10,return_mask=True)


plt.figure()
plt.plot(heading)
plt.plot(rxy)















    
    
    
    
    
    
    
    
    