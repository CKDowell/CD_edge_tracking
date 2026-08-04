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
import sklearn.linear_model as lm

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
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260429\f2\Trial1',# 17 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f1\Trial2', # 6 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260623\f1\Trial2', # 6 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260707\f1\Trial3' # 7 jumps
        ],
    
    "hDeltaC_FC2_EPG_ok_trackers": 
        # Be careful, since some flies are represented twice
        
        [
            r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260424\f1\Trial2', # Interesting data. FC2 and hDeltaC coming in and out of alignment a lot. Not a great tracker but lots of runs
            r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260425\f2\Trial3', # 3 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260401\f1\Trial2', # Some nice tracking 3 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260709\f1\Trial3', # Not many entries but FC2 is highly correlated with heading, largely ignoring hDeltaC
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260708\f2\Trial1', # one jump, nice neural data
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f2\Trial2',
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f1\Trial3', # No jumps but downwind noodling, could be good to look at hDC driving downwind walking
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f1\Trial2',# 1 jump. FC2 pops in and out of correlation with hDeltaC
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260618\f1\Trial1', # Amenotaxis with some downwind bouts. hDC and FC2 highly correlated. Interesting to see if downwind walking correlates with bump amp size
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260618\f1\Trial2', # Amenotaxis, FC2 signal looks like it drops during amenotaxis
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260617\f1\Trial2', # Anemotaxis all correlated
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f2\Trial3', # one jump nice transition between tracking and charging
        ]
    
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
        elif group_name=='hDeltaC_FC2_EPG_ok_trackers':
            etp = ET_paper(datadir,regions=['eb_ch1','fsb1_ch1','fsb2_ch2'])
        else:
            etp = ET_paper(datadir)
        all_flies[group_name][i] = etp
        
        
        
#%% Simple heuristics 
# Summary finding: correlation decreases to chance after odour on average, but this is not what we know to be true

plt.close('all')
minlen = 10 # minimum outside bout length
minlen = minlen*10 # multiply by timebase
dchoices = ['hDeltaC_FC2_EPG','hDeltaC_FC2_EPG_ok_trackers']
u = ug()

dlen = 0
# for di,d in enumerate(dchoices):
#     dlen += len(all_flies[d])
dlen = sum(len(all_flies[d]) for d in dchoices)
good_trackers = np.zeros(dlen,dtype='int');good_trackers[:len(all_flies[dchoices[0]])] = 1

bins = np.arange(0,30,1)
phase_data = np.zeros((len(bins),dlen,3))*np.nan
corr_data = np.zeros((len(bins),dlen,3))*np.nan



cnt = -1
jumpcounts = np.zeros(dlen)
for di,d in enumerate(dchoices):
    data = all_flies[d]
    for f in data:
        cnt +=1
        # Gather data
        etp = data[f]
        cxa = etp.cxa
        jumps = cxa.get_entries_exits_like_jumps()
        
        jlen = jumps[:,2]-jumps[:,1]
        jdx = jlen>=minlen
        
        jumps = jumps[jdx,:]
        jumpcounts[cnt] = len(jumps)
        # Phases
        hdcp = cxa.pdat['phase_fsb2_ch2']
        fc2p = cxa.pdat['phase_fsb1_ch1']
        pdiff = np.abs(ug.circ_subtract(hdcp,fc2p))
        
        # Wedges
        hdcw = cxa.pdat['wedges_fsb2_ch2']
        fc2w = cxa.pdat['wedges_fsb1_ch1']
        
        cvar = ug.rowwise_cov_smth(hdcw,fc2w,10)
        rs  =ug.rowwise_pearson_smth(hdcw,fc2w,10)
        
        # Get still/moving epochs
        x = cxa.ft2['ft_posx'];y = cxa.ft2['ft_posy']; t = np.arange(0,len(x))/10
        _,_,vel = u.get_velocity(x,y,t)
        stills = ug.get_still(.5,vel,minsize=5)
        
        
        # Iterate through entries and exits
        collatedphase = np.zeros((len(bins),len(jumps),3))*np.nan
        collatedcovar = np.zeros((len(bins),len(jumps),3))*np.nan
        for ji,j in enumerate(jumps):
            jdx = np.arange(j[1],j[2])
            tt = (jdx-jdx[0])/10
            tdx = tt<=np.max(bins)
            tstill = stills[jdx]
            movs = tstill==0
            
            
            
            tpdiff = pdiff[jdx[tdx]]
            bidx = np.digitize(tt[tdx],bins)-1
            ttstill = tstill[tdx]
            tcvar =cvar[jdx[tdx]]
            
            for b in np.unique(bidx):
                tpd = tpdiff[bidx==b]
                tcv = tcvar[bidx==b]
                ts = ttstill[bidx==b]
                
                # 1. Phase difference post odour
                collatedphase[b,ji,0] = stats.circmean(tpd,low=-np.pi,high=np.pi,nan_policy='omit')
                collatedphase[b,ji,1] = stats.circmean(tpd[ts>0],low=-np.pi,high=np.pi,nan_policy='omit')
                collatedphase[b,ji,2] = stats.circmean(tpd[ts==0],low=-np.pi,high=np.pi,nan_policy='omit')
                
                # 2. Correlation post odour
                collatedcovar[b,ji,0] = np.nanmean(tcv)
                collatedcovar[b,ji,1] = np.nanmean(tcv[ts>0])
                collatedcovar[b,ji,2] = np.nanmean(tcv[ts==0])
                
        phase_data[:,cnt,:] = stats.circmean(collatedphase,axis=1,low=-np.pi,nan_policy='omit')
        corr_data[:,cnt,:] = np.nanmean(collatedcovar,axis=1)
            # 2. Split that correlation by locomotor vs non locomotor state

colours = uplt.columnar_colours()
# Plot for good trackers
# Phase difference
titles = ['all','still','moving']
for i in range(3):
    plt.figure()
    plt.subplot(1,2,1)
    plt.title(titles[i]+ ' phase')
    pdata = phase_data[:,good_trackers==1,i]
    pmean = stats.circmean(pdata,axis=1,low=-np.pi,high=np.pi,nan_policy='omit')
    binarray = np.tile(bins[:,np.newaxis],6)
    plt.plot(binarray,180*pdata/np.pi,color=colours[1,:],alpha=.25)
    plt.plot(bins,180*pmean/np.pi,color=colours[1,:],linewidth=3)
    plt.ylabel('phase difference (deg)')
    plt.xlabel('time post odour (s)')
    #plt.scatter(binarray[:],pdata[:],color=colours[1,:],linewidth=0)
    
    # wedge covariance
    pdata = corr_data[:,good_trackers==1,i]
    pmean = np.nanmean(pdata,axis=1)
    plt.subplot(1,2,2)
    plt.title(titles[i]+ ' covariance')
    plt.plot(binarray,pdata,color=colours[1,:],alpha=.25)
    plt.plot(bins,pmean,color=colours[1,:],linewidth=3)
    plt.ylabel('covariance')
    plt.xlabel('time post odour (s)')

# Plot for OK trackers
#%% Example covariance and phase difference plot
choice = 'hDeltaC_FC2_EPG'
fly = 2
etp = data = all_flies[choice][fly]
cxa = etp.cxa
hdcw = cxa.pdat['wedges_fsb2_ch2']
fc2w = cxa.pdat['wedges_fsb1_ch1']

hdcp = cxa.pdat['offset_fsb2_ch2_phase'].to_numpy()
fc2p = cxa.pdat['offset_fsb1_ch1_phase'].to_numpy()
epgp = cxa.pdat['offset_eb_ch1_phase'].to_numpy()
ins = cxa.ft2['instrip'].to_numpy().astype(float)
pva = ug.get_pvas(fc2w)
cvar = ug.rowwise_cov_smth(hdcw,fc2w,10)
tt =  np.arange(0,len(epgp))/10
colours = uplt.columnar_colours()
savedir = r'Y:\Presentations\2026\08_LabMeeting\Panels'
x = np.arange(0,16)
offset = 0
example_covars = np.arange(-0.01,.04,.01)
plt.close('all')
xticks = []
with plt.rc_context({'font.size': 14}):
    plt.figure(figsize=(14,4))
    
    plt.scatter(tt,fc2p,s=10,linewidth=0,color=colours[2,:])
    plt.scatter(tt,hdcp,s=10,linewidth=0,color=colours[1,:])
    plt.fill_between(tt,ins-np.pi-1.25,ins*0-np.pi-1.25,color='r',linewidth=0)
    plt.plot(tt,cvar*50-np.pi-3,color='k')
    plt.plot(tt,cvar*00-np.pi-3,color='k',linestyle='--')
    plt.xlim([1150,1400])
    plt.yticks([-np.pi-3,-np.pi-3+.01*50,-np.pi-3+.02*50,-np.pi,-np.pi/2,0,np.pi/2,np.pi],labels=[0,.01,0.02,-180,-90,0,90,180])
    plt.xlabel('time (s)')
    plt.savefig(os.path.join(savedir,'ExamplePhaseAndCovariance.pdf'))
    plt.figure()
    for i,e in enumerate(example_covars[:-1]):
        dx = np.where((cvar>e)&(cvar<example_covars[i+1])&(ins==0))[0]
        
        r =dx[ np.random.randint(0,len(dx))]
        thdc = hdcw[r,:]
        tfc2 = fc2w[r,:]
        plt.plot(x+offset,thdc,color=colours[1,:],linewidth=2)
        plt.plot(x+offset,tfc2,color=colours[2,:],linewidth=2)
        xticks.append(offset+7.5)
        offset += 18
    plt.xticks(xticks,labels = ['-0.01 - 0 ','0 - 0.01','0.1 - 0.2','0.2 - 0.3'
        ])
    plt.xlabel('covariance')
    plt.ylabel('dF/F0')
#%% Plot trajectory snippets with high correlation between hDeltaC and FC2
from scipy.ndimage import median_filter
plt.close('all')
dchoices = ['hDeltaC_FC2_EPG']#,'hDeltaC_FC2_EPG_ok_trackers']
minsize = 30
# rank by FC2 mean PVA and see what you get
interptime = 50
dlen = sum(len(all_flies[d]) for d in dchoices)
bins = np.arange(0.05,0.25,.025)
outarray = np.zeros((interptime*3,2,dlen,len(bins)))
outphase = np.zeros((dlen,3,len(bins)))

out_proj = np.empty((1,6))
out_proj_control = np.empty((1,6))
# control is a random sample of trajectories 
outarray_control = np.zeros((interptime*3,2,dlen,len(bins)))
outphase_control = np.zeros((dlen,3,len(bins)))

cnt = -1

consample = 15 # ten random samples from each bin
dist_thresh = 5

for di,d in enumerate(dchoices):
    data = all_flies[d]
    for f in data:
        cnt = cnt+1
        plt.figure(f)
        etp = data[f]
        cxa= etp.cxa
        hdcw = cxa.pdat['wedges_fsb2_ch2']
        fc2w = cxa.pdat['wedges_fsb1_ch1']
        
        hdcp = cxa.pdat['offset_fsb2_ch2_phase'].to_numpy()
        fc2p = cxa.pdat['offset_fsb1_ch1_phase'].to_numpy()
        epgp = cxa.pdat['offset_eb_ch1_phase'].to_numpy()
        
        
        
        pva = ug.get_pvas(fc2w)
        #pva = (pva-np.mean(pva))/np.std(pva)
        ins = cxa.ft2['instrip'].to_numpy()
        ins = np.roll(ins,5) # account for MFC delay
        ins = np.array([np.roll(ins,i) for i in range(20)]) #extend instrip to account for exit correlation
        ins = np.sum(ins,axis=0)
        #ins = np.roll(ins,5) + np.roll(ins,10)+np.roll(ins,15)+np.roll(ins,20)
        
        ins  = ins>0
        x = cxa.ft2['ft_posx'].to_numpy();y = cxa.ft2['ft_posy'].to_numpy();
        x,y = cxa.fictrac_repair(x,y)
        cvar = ug.rowwise_cov_smth(hdcw,fc2w,10)
        blocks,bsize = ug.find_blocks(cvar>.01,mergeblocks=True,merg_threshold=3)
        bdx = bsize>=minsize
        blocks = blocks[bdx]
        bsize = bsize[bdx]
        bstr1 = ins[blocks]
        blocks = blocks[bstr1==0]
        bsize = bsize[bstr1==0]
        
        cvarlogical = ug.block2array(blocks,bsize,len(ins))
        
        
        
        
        newblocks = np.zeros(len(cvar))
        offset = 0
        
        pvmns = [np.median(pva[b:b+bsize[ib]]) for ib,b in enumerate(blocks)]
        pvs = np.argsort(pvmns)
        
        
        #for ib,b in enumerate(blocks):
        for i,ib in enumerate(pvs):
            b = blocks[ib]
            newblocks[b:b+bsize[ib]] = 1
            tx = x[b:b+bsize[ib]]
            ty = y[b:b+bsize[ib]]
            stx = tx[0]
            sty = ty[0]
            tx = tx-tx[0]
            ty = ty-ty[0]
            
            bx = x[max(b-50,0):b+1]-stx;by = y[max(b-50,0):b+1]-sty
            ax = x[b+bsize[ib]-1:b+bsize[ib]*2]-stx;ay = y[b+bsize[ib]-1:b+bsize[ib]*2]-sty;
            plt.plot(ax+offset,ay,color='b')
            #print(np.max(tx))
            plt.plot(tx+offset,ty,color='r')
            plt.plot(bx+offset,by,color='k')
            plt.text(offset,100,str(np.round(pvmns[ib],decimals=2)),fontsize=5)
            offset += np.max(tx)-np.min(tx)+10
            
        for bni,bn in enumerate(bins):
            if bni==len(bins)-1:
                bdx = pvmns>bn
            else:
                bdx = np.logical_and(pvmns>bn,pvmns<bins[bni+1])
            
            bdx = np.where(bdx)[0]
            
            trajmini = np.zeros((interptime*3,2,len(bdx)))*np.nan
            phasemini = np.zeros((len(bdx),3))*np.nan
            for i,ib in enumerate(bdx):
                b = blocks[ib]
                
                
                
                tx = x[b:b+bsize[ib]]
                ty = y[b:b+bsize[ib]]
                
                
                
                stx = tx[0]
                sty = ty[0]
                tx = tx-tx[0]
                ty = ty-ty[0]
                
                
                
                dx = np.diff(tx);dy = np.diff(ty);
                dist = np.sum(np.sqrt(dx**2+dy**2))
                if dist<dist_thresh:
                    continue
                
                phasemini[i,0] = stats.circmean(epgp[b:b+bsize[ib]],low=-np.pi,high=np.pi)
                phasemini[i,1] = stats.circmean(hdcp[b:b+bsize[ib]],low=-np.pi,high=np.pi)
                phasemini[i,2] = stats.circmean(fc2p[b:b+bsize[ib]],low=-np.pi,high=np.pi)
                
                projarray = np.array([tx[-1],ty[-1],tx[-1]/ty[-1],np.median(pva[b:b+bsize[ib]]),np.median(cvar[b:b+bsize[ib]]),f])[np.newaxis,:]
                
                #projarray = np.array([np.mean(dx),np.mean(dy),tx[-1]/ty[-1],np.median(pva[b:b+bsize[ib]]),np.median(cvar[b:b+bsize[ib]]),f])[np.newaxis,:]
                
                
                out_proj = np.append(out_proj,projarray,axis=0)
                
                bx = x[max(b-bsize[ib],0):b+1]-stx;by = y[max(b-bsize[ib],0):b+1]-sty
                ax = x[b+bsize[ib]-1:b+bsize[ib]*2]-stx;ay = y[b+bsize[ib]-1:b+bsize[ib]*2]-sty;
                
                oldtime = np.arange(0,len(bx))
                newtime = np.linspace(0,len(bx),interptime)
                trajmini[:interptime,0,i] = np.interp(newtime,oldtime,bx)
                trajmini[:interptime,1,i] = np.interp(newtime,oldtime,by)
                
                oldtime = np.arange(0,len(tx))
                newtime = np.linspace(0,len(tx),interptime)
                trajmini[interptime:interptime*2,0,i] = np.interp(newtime,oldtime,tx)
                trajmini[interptime:interptime*2,1,i] = np.interp(newtime,oldtime,ty)
                
                oldtime = np.arange(0,len(ax))
                newtime = np.linspace(0,len(ax),interptime)
                trajmini[-interptime:,0,i] = np.interp(newtime,oldtime,ax)
                trajmini[-interptime:,1,i] = np.interp(newtime,oldtime,ay)
                
            outarray[:,:,cnt,bni] = np.nanmean(trajmini,axis=2)
            outphase[cnt,:,bni] = stats.circmean(phasemini,axis=0,low=-np.pi,high=np.pi,nan_policy='omit')
        plt.gca().set_aspect('equal')
        # control blocks = 
        bsizemn = np.median(bsize).astype(int)
        pvam = median_filter(pva,int(bsizemn/2),mode='constant') # I think binsize is probably a little too restrictive
        #pvam = pva
        offset= 0
        plt.figure(f+100);plt.gca().set_aspect('equal')
        for bni,bn in enumerate(bins):
            if bni==len(bins)-1:
        
                notin = (ins==0)&(pvam>bn)&(cvarlogical==0)
            else:
                notin =(ins==0)&(pvam>bn)&(pvam<bins[bni+1])&(cvarlogical==0)
                
            notinb,nbs = ug.find_blocks(notin,mergeblocks=True,merg_threshold=minsize)
            nbsx = nbs>=minsize;nbs = nbs[nbsx];notinb = notinb[nbsx]
            
            print(bn,len(notinb))
            if len(notinb)>consample:
                notinb =notinb[ np.random.choice(len(notinb),consample,replace=False)]
            trajmini = np.zeros((interptime*3,2,len(notinb)))*np.nan
            phasemini = np.zeros((len(notinb),3))*np.nan
            for i,b in enumerate(notinb):
                #b = blocks[ib]
                
                
                tx = x[b:b+nbs[i]]
                ty = y[b:b+nbs[i]]
                stx = tx[0]
                sty = ty[0]
                tx = tx-tx[0]
                ty = ty-ty[0]
                
                
                
                
                
                dx = np.diff(tx);dy = np.diff(ty);
                dist = np.sum(np.sqrt(dx**2+dy**2))
                if dist<dist_thresh:
                    continue
                
                
                phasemini[i,0] = stats.circmean(epgp[b:b+nbs[i]],low=-np.pi,high=np.pi)
                phasemini[i,1] = stats.circmean(hdcp[b:b+nbs[i]],low=-np.pi,high=np.pi)
                phasemini[i,2] = stats.circmean(fc2p[b:b+nbs[i]],low=-np.pi,high=np.pi)
                
                projarray = np.array([tx[-1],ty[-1],tx[-1]/ty[-1],np.median(pva[b:b+nbs[i]]),np.median(cvar[b:b+nbs[i]]),f])[np.newaxis,:]
                #projarray = np.array([np.mean(dx),np.mean(dy),tx[-1]/ty[-1],np.median(pva[b:b+bsize[ib]]),np.median(cvar[b:b+bsize[ib]]),f])[np.newaxis,:]
                out_proj_control = np.append(out_proj_control,projarray,axis=0)
                
                bx = x[max(b-nbs[i],0):b+1]-stx;by = y[max(b-nbs[i],0):b+1]-sty
                ax = x[b+nbs[i]-1:b+nbs[i]*2]-stx;ay = y[b+nbs[i]-1:b+nbs[i]*2]-sty;
                
                oldtime = np.arange(0,len(bx))
                newtime = np.linspace(0,len(bx),interptime)
                trajmini[:interptime,0,i] = np.interp(newtime,oldtime,bx)
                trajmini[:interptime,1,i] = np.interp(newtime,oldtime,by)
                
                oldtime = np.arange(0,len(tx))
                newtime = np.linspace(0,len(tx),interptime)
                trajmini[interptime:interptime*2,0,i] = np.interp(newtime,oldtime,tx)
                trajmini[interptime:interptime*2,1,i] = np.interp(newtime,oldtime,ty)
                
                oldtime = np.arange(0,len(ax))
                newtime = np.linspace(0,len(ax),interptime)
                trajmini[-interptime:,0,i] = np.interp(newtime,oldtime,ax)
                trajmini[-interptime:,1,i] = np.interp(newtime,oldtime,ay)
                
                plt.plot(ax+offset,ay,color='b')
                #print(np.max(tx))
                plt.plot(tx+offset,ty,color='r')
                plt.plot(bx+offset,by,color='k')
               # plt.text(offset,100,str(np.round(pvmns[ib],decimals=2)),fontsize=5)
                offset += np.max(tx)-np.min(tx)+10
                
            outarray_control[:,:,cnt,bni] = np.nanmean(trajmini,2)    
            outphase_control[cnt,:,bni] = stats.circmean(phasemini,axis=0,low=-np.pi,high=np.pi,nan_policy='omit')
        
        
        
out_proj = out_proj[1:,:]
out_proj_control = out_proj_control[1:,:]    
            
        # plt.figure()
        # plt.plot(cvar>0.015)
        # plt.plot(newblocks,color='g')
        # plt.plot(ins*.2,color='r')
plt.figure()
#plt.scatter(out_proj[:,2],np.abs(out_proj[:,0]),s=30,c=out_proj[:,3]) 
plt.subplot(1,2,1)
c =        out_proj[:,3]
c[c>.2] = .2
plt.scatter(out_proj[:,0],out_proj[:,1],s=10,c=c)
plt.xlim([-100,100])
plt.ylim([-100,300])
plt.gca().set_aspect('equal')
plt.subplot(1,2,2)
c =        out_proj_control[:,3]
c[c>.2] = .2
plt.scatter(out_proj_control[:,0],out_proj_control[:,1],s=10,c=c)
plt.xlim([-100,100])
plt.ylim([-100,300])
plt.gca().set_aspect('equal')

plt.figure()
c =        out_proj_control[:,3]
plt.scatter(out_proj_control[c>.125,0],out_proj_control[c>.125,1],s=30,color=[.7,.7,.7])
plt.scatter(np.mean(out_proj_control[c>.125,0]),np.mean(out_proj_control[c>.125,1]),s=30,color='k')
c =        out_proj[:,3]
plt.scatter(out_proj[c>.125,0],out_proj[c>.125,1],s=30,color='r')
plt.scatter(np.mean(out_proj[c>.125,0]),np.mean(out_proj[c>.125,1]),color=[1,.8,.8])


plt.figure()
plt.subplot(1,2,1)
c1 =        out_proj_control[:,3]
plt.scatter(out_proj_control[c1>.125,0],out_proj_control[c1>.125,1],s=30,c=out_proj_control[c1>.125,-1])
#plt.scatter(np.mean(out_proj_control[c>.125,0]),np.mean(out_proj_control[c>.125,1]),s=30,c=out_proj_control[c>.125,-1])
plt.xlim([-100,100])
plt.ylim([-100,300])
plt.subplot(1,2,2)
c2 =        out_proj[:,3]
plt.scatter(out_proj[c2>.125,0],out_proj[c2>.125,1],s=30,c=out_proj[c2>.125,-1])
plt.xlim([-100,100])
plt.ylim([-100,300])
#plt.scatter(np.mean(out_proj[c>.125,0]),np.mean(out_proj[c>.125,1]),color=[1,.8,.8])

stats.ttest_ind(out_proj[c2>.125,0],out_proj_control[c1>.125,0])
stats.ttest_ind(out_proj[c2>.125,1],out_proj_control[c1>.125,1])

#%%
data = out_proj[c2>.1,:]
datac = out_proj_control[c1>.1,:]
dat2 = np.zeros((2,6))
dat3 = np.zeros((2,6))
for f in range(6):
    dat2[:,f] = np.mean(data[data[:,-1]==f,:2],axis=0)
    dat3[:,f] = np.mean(datac[datac[:,-1]==f,:2],axis=0)
plt.scatter(dat2[0,:],dat2[1,:],color='r')
plt.scatter(dat3[0,:],dat3[1,:],color='k')

plt.plot([dat2[0,:],dat3[0,:]],[dat2[1,:],dat3[1,:]] ,color='k',zorder=-1)

#%% 
plt.scatter(out_proj[:,3],np.abs(out_proj[:,0]))
plt.scatter(out_proj_control[:,3],np.abs(out_proj_control[:,0]))
#%%
#plt.close('all')
gain =10
plt.figure();plt.gca().set_aspect('equal')    
colours = np.array([[0,0,0],[1,0,0],[0,0,1]])
colours2 =  uplt.columnar_colours()
colours2[0,:] = 0
for f in range(outarray.shape[2]):
    offset = 0
   # plt.figure();plt.gca().set_aspect('equal') 
    for b in range(outarray.shape[3]-1): # very little data in final bin
        offset +=30
        for i in range(3):
            tx = outarray[interptime*i:interptime*(i+1),0,f,b]
            ty = outarray[interptime*i:interptime*(i+1),1,f,b]
            plt.plot(tx+offset,ty,color=colours[i,:],alpha=.25)
            
            tp = outphase[f,i,b]
            px = np.array([0,np.sin(tp)])*gain;py = np.array([0,np.cos(tp)])*gain;
            plt.plot(px+offset,py-50,color=colours2[i,:],alpha=.25)
            
            
#plt.close('all')
#plt.figure();plt.gca().set_aspect('equal')    
colours = np.array([[0,0,0],[1,0,0],[0,0,1]])
for f in range(outarray.shape[2]):
    offset = 0
   # plt.figure();plt.gca().set_aspect('equal') 
    for b in range(outarray.shape[3]-1):
        offset +=30
        for i in range(3):
            tx = outarray_control[interptime*i:interptime*(i+1),0,f,b]
            ty = outarray_control[interptime*i:interptime*(i+1),1,f,b]
            plt.plot(tx+offset,ty+100,color=colours[i,:],alpha=.25)        
            tp = outphase_control[f,i,b]
            px = np.array([0,np.sin(tp)])*gain;py = np.array([0,np.cos(tp)])*gain;
            plt.plot(px+offset,py+50,color=colours2[i,:],alpha=.25)
            
            
pmean = np.nanmean(outarray,axis=2)
offset = 0
#plt.figure();plt.gca().set_aspect('equal') 
for b in range(outarray.shape[3]-1):
    offset +=30
    for i in range(3):
        tx = pmean[interptime*i:interptime*(i+1),0,b]
        ty = pmean[interptime*i:interptime*(i+1),1,b]
        plt.plot(tx+offset,ty,color=colours[i,:])
        
        
pmean = np.nanmean(outarray_control,axis=2)
offset = 0
xticks = np.array([])
for b in range(outarray.shape[3]-1):
    offset +=30
    xticks = np.append(xticks,offset)
    for i in range(3):
        tx = pmean[interptime*i:interptime*(i+1),0,b]
        ty = pmean[interptime*i:interptime*(i+1),1,b]
        plt.plot(tx+offset,ty+100,color=colours[i,:])


plt.plot([-25,-25],[-10,15],color='k')
plt.plot([-25,0],[-10,-10],color='k')
plt.text(-12.5,-18,'25 mm',horizontalalignment='center')
pltbins = bins[1:]-np.mean(np.diff(bins))/2
pltbins = np.round(pltbins,decimals=2)
plt.xticks(xticks,labels=pltbins)
plt.yticks([0,100],labels=['hDeltaC and FC2 coherent','FC2 and hDeltaC not coherent'])
plt.xlabel('FC2 PVA')
#%% Variation on above, except I break down the behaviour into movement bouts
# these can then be broken down further by the rdp algorithm
plt.close('all')
from rdp import rdp
dchoices = ['hDeltaC_FC2_EPG']#,'hDeltaC_FC2_EPG_ok_trackers']

angular_threshold = np.pi*(30/180)
datalist = []
cnt = 0
u = ug()
for di,d in enumerate(dchoices):
    data = all_flies[d]
    for f in data:
        cnt = cnt+1
        etp = data[f]
        cxa= etp.cxa
        hdcw = cxa.pdat['wedges_fsb2_ch2']
        fc2w = cxa.pdat['wedges_fsb1_ch1']
        
        hdcp = cxa.pdat['offset_fsb2_ch2_phase'].to_numpy()
        fc2p = cxa.pdat['offset_fsb1_ch1_phase'].to_numpy()
        epgp = cxa.pdat['offset_eb_ch1_phase'].to_numpy()
        
        
        
        pva = ug.get_pvas(fc2w)
        #pva = (pva-np.mean(pva))/np.std(pva)
        ins = cxa.ft2['instrip'].to_numpy();inso= ins.copy()
        ins = np.roll(ins,5) # account for MFC delay
        ins = np.array([np.roll(ins,i) for i in range(20)]) #extend instrip to account for exit correlation
        ins = np.sum(ins,axis=0)
        #ins = np.roll(ins,5) + np.roll(ins,10)+np.roll(ins,15)+np.roll(ins,20)
        
        ins  = ins>0
        
        
        
        x = cxa.ft2['ft_posx'].to_numpy();y = cxa.ft2['ft_posy'].to_numpy();
        x,y = cxa.fictrac_repair(x,y)
        t = np.arange(0,len(x))/10
        masterdx = np.arange(0,len(x))
        _,_,vel = u.get_velocity(x,y,t)
        still = ug.get_still(0.5,vel)
        mov = still==0
        
        cvar = ug.rowwise_cov_smth(hdcw,fc2w,10)
        cvar[cvar==0] = np.nan
        plt.figure()
        plt.gca().set_aspect('equal')
        blocks,bsize = ug.find_blocks(mov,mergeblocks=True,merg_threshold=5)
        
        for ib,b in enumerate(blocks):
            dx = np.arange(b,b+bsize[ib])
            tx = x[dx];ty = y[dx]
            plt.plot(tx,ty)
            
            txy = np.append(tx[:,np.newaxis],ty[:,np.newaxis],axis=1)
            msk = rdp(txy,epsilon=1,return_mask=True) # RDP
            
            rxy = txy[msk,:]
            plt.plot(rxy[:,0],rxy[:,1],color='r')
            if rxy.shape[0]>2:
                rvec = rxy[1:,:]-rxy[:-1,:]
                rvecj = rvec[:,1]+1j*rvec[:,0]
                rangle = np.angle(rvecj)
                dangle = np.abs(ug.circ_subtract(rangle[1:],rangle[:-1]))
                dkeep = np.where(dangle>angular_threshold)[0]+1 
                
                
                dkeep = np.append(0,dkeep);dkeep = np.append(dkeep,rxy.shape[0]-1)
                
                rxy = rxy[dkeep,:]
                
                rdx = dx[msk][dkeep]
                if len(dkeep)<len(dangle):
                    print('Merge occured')
            else:
                rdx = dx[msk]
            for ir,r in enumerate(rdx[:-1]):
                rvec = rxy[ir+1,:] -rxy[ir,:]
                rangle = np.angle(rvec[1]+1j*rvec[0])
                thisdx = np.arange(r,rdx[ir+1])
                datarray = np.array(
                [r,rdx[ir+1], # RDP index
                 (rdx[ir+1]-r)/10, # time of epoch
                 rangle, # RDP angle
                 rvec[0],
                 rvec[1],
                 np.nanmedian(cvar[thisdx]), # median covariance between hDeltaC and FC2
                 np.median(pva[thisdx]), # median PVA of FC2
                 stats.circmean(epgp[thisdx],low=-np.pi,high=np.pi), # Phase means
                 stats.circmean(hdcp[thisdx],low=-np.pi,high=np.pi),
                 stats.circmean(fc2p[thisdx],low=-np.pi,high=np.pi),
                 inso[thisdx[0]], #start instrip
                 inso[thisdx[-1]], # end instrip
                 np.sum(inso[thisdx])/len(thisdx), # percent in strip
                 np.mean(hdcw[thisdx,:]),
                 np.mean(fc2w[thisdx,:]),
                 di, # data origin
                    f # fly
                    ]
                )
                datalist.append(datarray)
        #         break
        #     break
        # break
    
    
            #plt.plot(rxy[:,0],rxy[:,1],color='r')
            
        #plt.scatter(x[ins>0],y[ins>0],color=[.5,.5,.7])
        #plt.scatter(x[inso>0],y[inso>0],color=[.7,.7,.7])
        # plt.plot(mov*np.max(vel))
        # plt.plot(vel)
data_out = np.array(datalist)
data_outpd = pd.DataFrame(data=data_out,columns = [
    'start','end','duration','rdp_angle','rdpx','rdpy','median_covar',
    'median_pvaFC2','epg_phasemean','hdc_phasemean','fc2_phasemean','strip_start','strip_end',
    'percent_strip','mean_col_hdc','mean_col_fc2','data_index','fly_index']
                          )
#%% Diagnostic plot for one fly
plt.close('all')
fly = 2
minduration = 1 
maxduration = 10
strip_percentage = .5
etp = all_flies[d][fly]
cxa =etp.cxa
x = cxa.ft2['ft_posx'].to_numpy();y = cxa.ft2['ft_posy'].to_numpy();x,y = cxa.fictrac_repair(x,y)
ins = cxa.ft2['instrip']
duration = data_outpd['duration'].to_numpy()
stripstart = data_outpd['strip_start'].to_numpy()
stripend = data_outpd['strip_end'].to_numpy()
percent_strip = data_outpd['percent_strip'].to_numpy()
flies = data_outpd['fly_index'].to_numpy()
fdx = flies==fly
keepdx = (duration>=minduration)&(stripstart==0)&(duration<=maxduration)&fdx&(percent_strip<strip_percentage)

rdx = data_outpd[['start','end']].to_numpy().astype(int)

plt.figure()
plt.plot(x,y,color='k')
rx = x[rdx[fdx]]
ry = y[rdx[fdx]]
plt.plot(rx.T,ry.T,color='b')

rx = x[rdx[keepdx]]
ry = y[rdx[keepdx]]
plt.plot(rx.T,ry.T,color='r')

plt.scatter(x[ins>0],y[ins>0],color=[.8,.8,.8])
plt.gca().set_aspect('equal')

#%% Clean data and plot
plt.close('all')
minduration = 1
maxduration = 10
strip_percentage = .5

duration = data_outpd['duration'].to_numpy()
stripstart = data_outpd['strip_start'].to_numpy()
stripend = data_outpd['strip_end'].to_numpy()
percent_strip = data_outpd['percent_strip'].to_numpy()
flies = data_outpd['fly_index']
projdat = []
projfly = []
projmeans = np.zeros((6,4))

bins = np.arange(0,6,.5)
binspva = np.arange(0.025,0.25,.025)
binscovar = np.arange(-.005,0.03,0.005)

projbins = np.zeros((6,len(bins)-1))
pvabins = np.zeros((6,len( binspva)-1))
covarbins = np.zeros((6,len(binscovar)-1))
phasebins = np.zeros((6,len(bins)-1,2))
for f in range(6):
    fdx = flies==f
    keepdx = (duration>=minduration)&(stripstart==0)&(duration<=maxduration)&fdx&(percent_strip<strip_percentage)
    plt.figure()
    x = data_outpd['rdpx'].to_numpy()[keepdx]
    y = data_outpd['rdpy'].to_numpy()[keepdx]
    c = data_outpd['median_covar'].to_numpy()[keepdx]
    
    plt.subplot(2,2,1)
    plt.scatter(c,np.abs(x),s=10)
    plt.xlabel('covar')
    plt.ylabel('x')
    
    plt.subplot(2,2,2)
    plt.scatter(c,np.abs(x)/np.abs(y))
    plt.yscale('log')
    plt.xlabel('covar');plt.ylabel('x/y')
    
    plt.subplot(2,2,3)
    c[c>.02] = .02
    c[c<0] = 0
    plt.scatter(x,y,c=c)
    plt.ylabel('y');plt.xlabel('x')
    plt.title('covar')
    
    c2 = data_outpd['median_pvaFC2'].to_numpy()[keepdx]
    c2[c2>.2] = .2 
    plt.subplot(2,2,4)
    plt.scatter(x,y,c=c2)
    plt.ylabel('y');plt.xlabel('x')

    plt.title('FC2 pva')
    
    rdpan = data_outpd['rdp_angle'].to_numpy()[keepdx][:,np.newaxis]
    phases = data_outpd[['hdc_phasemean','fc2_phasemean']].to_numpy()[keepdx]
    plt.figure()
    c = data_outpd['median_covar'].to_numpy()[keepdx][:,np.newaxis]
    rx = np.append(0*np.sin(rdpan),np.sin(rdpan),axis=1)*c
    ry  = np.append(0*np.sin(rdpan),np.cos(rdpan),axis=1)*c
    plt.plot(rx.T,ry.T,color='k',alpha=.5)
    plt.gca().set_aspect('equal')
    
    plt.figure()
    c2 = data_outpd['median_pvaFC2'].to_numpy()[keepdx]
    c = c.squeeze()
    
    reg = lm.LinearRegression()
    reg.fit_intercept = False
    xfit = c.reshape(-1, 1)
    yfit = c2.copy()-.05
    fdx = xfit>0
    xfit = xfit[fdx.squeeze(),:];yfit = yfit[fdx.squeeze()]
    xnorm = np.std(xfit);ynorm = np.std(yfit)
    xfit = xfit/xnorm;yfit = yfit/ynorm
    
    reg.fit(xfit,yfit)
    xp = np.array([0,.02])
    yp = reg.predict(xp.reshape(-1,1))
    plt.plot(xp,yp,color='b')
    plt.scatter(xfit,yfit,c=np.abs(rdpan[fdx.squeeze(),:]),cmap = 'cividis')
    proj_vec = np.array([1,reg.coef_[0]]);proj_vec = proj_vec/np.sqrt(np.sum(proj_vec**2))
    
    Cmat = np.append(c[:,np.newaxis],c2[:,np.newaxis]-.05,axis=1)
    Cmat[:,0] = Cmat[:,0]/xnorm;Cmat[:,1] = Cmat[:,1]/ynorm
    cproj = np.matmul(Cmat,proj_vec)
    
    
    plt.figure()
    plt.scatter(cproj,np.abs(rdpan))
    tproj = np.append(cproj[:,np.newaxis], rdpan, axis=1)   # (N_f, 2): [cproj, rdp_angle]
    projdat.append(tproj)
    projfly.append(np.full(cproj.shape[0], f))
    
    projmeans[f,0] = stats.circmean(np.abs(rdpan[cproj<np.percentile(cproj,75)]),low=-np.pi,high=np.pi); 
    projmeans[f,1] = stats.circmean(np.abs(rdpan[cproj>np.percentile(cproj,75)]),low=-np.pi,high=np.pi);
    
    projbins[f,:] = stats.binned_statistic(
    cproj, np.abs(rdpan).squeeze(),
    statistic=lambda v: stats.circmean(v, low=-np.pi, high=np.pi) if len(v) else np.nan,
    bins=bins
    )[0]
    
    
    pvabins[f,:] = stats.binned_statistic(
        c2, np.abs(rdpan).squeeze(),
        statistic=lambda v: stats.circmean(v,low=-np.pi,high=np.pi) if len(v) else np.nan,
        bins = binspva
        )[0]
    
    covarbins[f,:] = stats.binned_statistic(
        c, np.abs(rdpan).squeeze(),
        statistic=lambda v: stats.circmean(v,low=-np.pi,high=np.pi) if len(v) else np.nan,
        bins = binscovar
        )[0]
    
    for pi in range(2):
        phasebins[f,:,pi] = stats.binned_statistic(
        cproj, np.abs(phases[:,pi]).squeeze(),
        statistic=lambda v: stats.circmean(v, low=-np.pi, high=np.pi) if len(v) else np.nan,
        bins=bins
        )[0]
    
    
    
projdat = np.concatenate(projdat, axis=0)   # (sum(N_f), 2)
projfly = np.concatenate(projfly)
keepdx = (duration>=minduration)&(stripstart==0)&(duration<=maxduration)&(percent_strip<strip_percentage)
pvacovar = data_outpd[['median_pvaFC2','median_covar']].to_numpy()[keepdx]
phases  = data_outpd[['hdc_phasemean','fc2_phasemean']].to_numpy()[keepdx]
keepdata = data_outpd[keepdx]

#%% Regression fits and plots for covariance of factors
plt.close('all')
plt.figure(1,figsize=(10,6.2))
plt.figure(2,figsize=(10,6.2))
for f in range(6):
    fdx = flies==f
    keepdx = (duration>=minduration)&(stripstart==0)&(duration<=maxduration)&fdx&(percent_strip<strip_percentage)
    c = data_outpd['median_covar'].to_numpy()[keepdx]
    c2 = data_outpd['median_pvaFC2'].to_numpy()[keepdx]
    rdpan = data_outpd['rdp_angle'].to_numpy()[keepdx]
    
    xfit = c.reshape(-1, 1)
    yfit = c2.copy()
    fdx = xfit>0
    xfit = xfit[fdx.squeeze(),:];yfit = yfit[fdx.squeeze()]
    reg = lm.LinearRegression()
    
    reg.fit_intercept = False
    yfit = yfit-.05
    reg.fit(xfit,yfit)
    xp = np.array([0,.025])
    yp = reg.predict(xp.reshape(-1,1))+.05
    plt.figure(1)
    plt.subplot(2,3,f+1)
    plt.plot(xp,yp,color='r')
    I = np.argsort(np.abs(rdpan))
    plt.scatter(c,c2,c=np.abs(rdpan),cmap='coolwarm',s=15)
    plt.xlim([-.015,.03])
    plt.ylim([0,.26])
    
    plt.figure(2)
    plt.subplot(2,3,f+1)
    plt.plot(xp,yp,color='r')
    plt.scatter(c,c2,color='k',s=15)
    #plt.title(f)
    plt.ylim([0,.26])
    plt.xlim([-.015,.03])

savedir =r'Y:\Data\FCI\FCI_summaries\FC2_hDeltaC_comparison'
plt.figure(1)
plt.savefig(os.path.join(savedir,'PVA_Coherence_Correlation_simple.pdf'))
plt.figure(2)
plt.savefig(os.path.join(savedir,'PVA_Coherence_Correlation_colour.pdf'))
#%% Plot projection figure phew, that was a lot of work
plt.close('all')
colours = uplt.columnar_colours()
savedir = r'Y:\Data\FCI\FCI_summaries\FC2_hDeltaC_comparison'
sname = 'FC2_hDeltaC_coherence_PVA_proj_vs_RDP_heading'
pnames = ['hDeltaC','FC2']
with plt.rc_context({'font.size': 14}):
    plt.figure()
    
    plt.scatter(180*projmeans[:,0]/np.pi,180*projmeans[:,1]/np.pi,color='k')
    plt.xlim([0,180])
    plt.ylim([0,180])
    plt.plot([0,180],[0,180],color='k')
    plt.plot([90,90],[0,180],color='r',linestyle='--')
    plt.plot([0,180],[90,90],color='r',linestyle='--')
    plt.xlabel('direction low projection (deg)')
    plt.ylabel('direction high projection (deg)')
    plt.xticks(np.arange(0,225,45))
    plt.yticks(np.arange(0,225,45))
    plt.savefig(os.path.join(savedir,sname+'.png'))
    plt.savefig(os.path.join(savedir,sname+'.pdf'))
    
    plt.figure()
    plt.scatter(projdat[:,0],180*np.abs(projdat[:,1])/np.pi,color='k',s=20,alpha=.25)
    #plt.plot([.05,.05],[0,180],color='r',linestyle='--')
    pltbins = bins[1:]-np.diff(bins)/2
    pltbins = np.tile(pltbins[:,np.newaxis],6)
    plt.plot(pltbins,180*projbins.T/np.pi,color='r',alpha=.75,linestyle='--')
    plt.plot(pltbins[:,0],180*stats.circmean(projbins,axis=0,low=-np.pi,high=np.pi,nan_policy='omit')/np.pi,linewidth=3,color='r')
    plt.ylabel('trajectory direction (deg)')
    plt.xlabel('projection')
    plt.ylim([0,180])
    plt.yticks(np.arange(0,225,45))
    plt.savefig(os.path.join(savedir,'projection_summary'+'.png'))
    plt.savefig(os.path.join(savedir,'projection_summary'+'.pdf'))
    
    
    phasebinmean = stats.circmean(phasebins,axis=0,low=-np.pi,high=np.pi,nan_policy='omit')
    for pi in range(2):
        plt.figure()
        plt.scatter(projdat[:,0],180*np.abs(phases[:,pi])/np.pi,color='k',s=20,alpha=.25)
        plt.plot(pltbins,180*phasebins[:,:,pi].T/np.pi,color=colours[1+pi,:],alpha=.75,linestyle='--')
        plt.plot(pltbins[:,0],180*phasebinmean[:,pi]/np.pi,color=colours[1+pi,:],linestyle='-',linewidth=3)
        plt.ylabel('phase (deg)')
        plt.xlabel('projection')
        plt.ylim([0,180])
        plt.yticks(np.arange(0,225,45))
        plt.savefig(os.path.join(savedir,'projection_summary_phase'+pnames[pi]+'.png'))
        plt.savefig(os.path.join(savedir,'projection_summary_phase'+pnames[pi]+'.pdf'))
    
    plt.figure()
    plt.scatter(pvacovar[:,1],180*np.abs(projdat[:,1])/np.pi,color='k',s=20,alpha=.25)
    pltbins = binscovar[1:]-np.diff(binscovar)/2
    pltbins = np.tile(pltbins[:,np.newaxis],6)
    plt.plot(pltbins,180*covarbins.T/np.pi,color='r',alpha=.75,linestyle='--')
    plt.plot(pltbins[:,0],180*stats.circmean(covarbins,axis=0,low=-np.pi,high=np.pi,nan_policy='omit')/np.pi,linewidth=3,color='r')
    plt.ylabel('trajectory direction (deg)')
    plt.xlabel('covariance')
    plt.xlim([-.02,.04]) # one outlier skewing plot
    plt.ylim([0,180])
    plt.yticks(np.arange(0,225,45))
    plt.savefig(os.path.join(savedir,'covariance_summary'+'.png'))
    plt.savefig(os.path.join(savedir,'covariance_summary'+'.pdf'))
    
    
    plt.figure()
    plt.scatter(pvacovar[:,0],180*np.abs(projdat[:,1])/np.pi,color='k',s=20,alpha=.25)
    pltbins = binspva[1:]-np.diff(binspva)/2
    pltbins = np.tile(pltbins[:,np.newaxis],6)
    plt.plot(pltbins,180*pvabins.T/np.pi,color='r',alpha=.75,linestyle='--')
    plt.plot(pltbins[:,0],180*stats.circmean(pvabins,axis=0,low=-np.pi,high=np.pi,nan_policy='omit')/np.pi,linewidth=3,color='r')
    plt.ylabel('trajectory direction (deg)')
    plt.xlabel('FC2 pva')
    plt.ylim([0,180])
    plt.yticks(np.arange(0,225,45))
    plt.savefig(os.path.join(savedir,'pvaFC2_summary'+'.png'))
    plt.savefig(os.path.join(savedir,'pvaFC2_summary'+'.pdf'))
    
#%%  plot fly trajectories binned by projection
plt.close('all')
projection_threshold = 2 
tbins = 50
trajectories = np.zeros((tbins*2,2,6,2))
before_time = 1 
colours,tcmap = uplt.romaO_cmap( 100)
crange = np.linspace(0,4,100)
crange = np.linspace(-np.pi,np.pi,100)
xdx = np.linspace(0,10,10000)
xrange = np.linspace(0,100,10000)
ydx = np.linspace(-np.pi,np.pi,10000)
yrange = np.linspace(0,50,10000)
savedir = r'Y:\Data\FCI\FCI_summaries\FC2_hDeltaC_comparison'
for f in range(6):
    etp  = all_flies['hDeltaC_FC2_EPG'][f]
    cxa = etp.cxa
    
    x = cxa.ft2['ft_posx'].to_numpy();y= cxa.ft2['ft_posy'].to_numpy()
    x,y = cxa.fictrac_repair(x,y)
    plt.figure(figsize= (15,10))
    plt.title(str(f))
    plt.gca().set_aspect('equal')
    offset = 0
    #for t in range(2):
        # if t==0:
        #     threshdx = projdat[:,0]<projection_threshold
        # else:
        #     threshdx = projdat[:,0]>projection_threshold
    
    
    fdx = (keepdata['fly_index'].to_numpy()==f)#&threshdx
    start_end = keepdata[['start','end']].to_numpy()[fdx,:].astype(int)
    tproj = projdat[fdx,:]
    tproj[:,1] = tproj[:,1]
    trajmini = np.zeros((tbins*2,2,start_end.shape[0],2))
    I = np.argsort(tproj[:,0])
    tproj = tproj[I,:]
    start_end = start_end[I,:]
    
    for i,s in enumerate(start_end):
        dx = np.arange(s[0],s[1])
        dxbef = np.arange(s[0]-10,s[0]+1)
        tx = x[dx];xnrm = tx[0];tx = tx-xnrm;
        tx = tx
        ty = y[dx];ynrm = ty[0];ty = ty-ynrm;
        d = np.sqrt((tx[-1]-tx[0])**2+(tx[-1]-ty[0])**2)
        d = max(d,1)
        
        # tx = tx/d
        # ty = ty/2
        
        cdx = ug.find_nearest(tproj[i,1],crange)
        offset = xrange[ug.find_nearest(tproj[i,0],xdx)]
        offsety = yrange[ug.find_nearest(tproj[i,1],ydx)]
        plt.plot(tx+offset,ty,color=colours[cdx,:],alpha=.75,zorder=-1)
        plt.plot([tx[0]+offset,tx[0]+offset],[ty[0]-40,ty[0]-25],color=colours[cdx],zorder=10,linewidth=1)
        #plt.plot(tx+offset,ty,color='k',linewidth=.5,alpha=.5)
        #offset += 1
    
    plt.xlim([-50,80])
    plt.ylim([-50,150])
    plt.savefig(os.path.join(savedir,'Proj_trajectories_'+str(f)+'.pdf'))
    
    
plt.figure()
plt.imshow(np.zeros((10,10)),cmap=tcmap,vmin=-180,vmax=180)
plt.colorbar(label = 'trajectory direction (degrees)',ticks=np.arange(-180,270,90))
plt.savefig(os.path.join(r'Y:\Presentations\StockImages','romaO_colourbar.pdf'))
#%% Plot coherence versus offset in FC2/EPG

# meanactivity = data_outpd[['mean_col_hdc','mean_col_fc2','median_pvaFC2',
#                            'epg_phasemean','hdc_phasemean','fc2_phasemean']].to_numpy()
flies = data_outpd['fly_index']
binspva = np.arange(0.025,0.25,.025)
binsdf = np.arange(0,.9,.1)
pvamn = np.zeros((6,len(binspva)-1))
mnmn = np.zeros((6,len(binsdf)-1))

pvamn_epg = np.zeros((6,len(binspva)-1))
mnmn_epg = np.zeros((6,len(binsdf)-1))

savedir = r'Y:\Data\FCI\FCI_summaries\FC2_hDeltaC_comparison'
plt.close('all')
for f in range(6):
    fdx = flies==f
    keepdx = (duration>=minduration)&(stripstart==0)&(duration<=maxduration)&(percent_strip<strip_percentage)&fdx
    
    epg = data_outpd['epg_phasemean'].to_numpy()[keepdx]
    fc2 =  data_outpd['fc2_phasemean'].to_numpy()[keepdx]
    pdiff = np.abs(ug.circ_subtract(fc2,epg))
    medw = data_outpd['mean_col_fc2'].to_numpy()[keepdx]
    pva = data_outpd['median_pvaFC2'].to_numpy()[keepdx]
    
    pvamn[f,:] = stats.binned_statistic(
        pva, np.abs(pdiff).squeeze(),
        statistic=lambda v: stats.circmean(v,low=-np.pi,high=np.pi) if len(v) else np.nan,
        bins = binspva
        )[0]
    
    mnmn[f,:] = stats.binned_statistic(
        medw, np.abs(pdiff).squeeze(),
        statistic=lambda v: stats.circmean(v,low=-np.pi,high=np.pi) if len(v) else np.nan,
        bins = binsdf
        )[0]

    pvamn_epg[f,:] = stats.binned_statistic(
        pva, np.abs(epg).squeeze(),
        statistic=lambda v: stats.circmean(v,low=-np.pi,high=np.pi) if len(v) else np.nan,
        bins = binspva
        )[0]
    
    mnmn_epg[f,:] = stats.binned_statistic(
        medw, np.abs(epg).squeeze(),
        statistic=lambda v: stats.circmean(v,low=-np.pi,high=np.pi) if len(v) else np.nan,
        bins = binsdf
        )[0]


keepdx = (duration>=minduration)&(stripstart==0)&(duration<=maxduration)&(percent_strip<strip_percentage)

epg = data_outpd['epg_phasemean'].to_numpy()[keepdx]
fc2 =  data_outpd['fc2_phasemean'].to_numpy()[keepdx]
pdiff = np.abs(ug.circ_subtract(fc2,epg))
medw = data_outpd['mean_col_fc2'].to_numpy()[keepdx]
pva = data_outpd['median_pvaFC2'].to_numpy()[keepdx]

with plt.rc_context({'font.size': 14}):
    
    plt.figure()
    plt.scatter(pva,180*np.abs(pdiff)/np.pi,color='k',s=20,alpha=.25)
    pltbins = binspva[1:]-np.diff(binspva)/2
    pltbins = np.tile(pltbins[:,np.newaxis],6)
    plt.plot(pltbins,180*pvamn.T/np.pi,color='r',alpha=.75,linestyle='--')
    plt.plot(pltbins[:,0],180*stats.circmean(pvamn,axis=0,low=-np.pi,high=np.pi,nan_policy='omit')/np.pi,linewidth=3,color='r')
    plt.ylabel('FC2-EPG offset (deg)')
    plt.xlabel('FC2 pva')
    plt.ylim([0,180])
    plt.yticks(np.arange(0,225,45))
    plt.savefig(os.path.join(savedir,'PVA_vs_FC2_EPG_offset'+'.png'))
    plt.savefig(os.path.join(savedir,'PVA_vs_FC2_EPG_offset'+'.pdf'))


    plt.figure()
    plt.scatter(medw,180*np.abs(pdiff)/np.pi,color='k',s=20,alpha=.25)
    pltbins = binsdf[1:]-np.diff(binsdf)/2
    pltbins = np.tile(pltbins[:,np.newaxis],6)
    plt.plot(pltbins,180*mnmn.T/np.pi,color='r',alpha=.75,linestyle='--')
    plt.plot(pltbins[:,0],180*stats.circmean(mnmn,axis=0,low=-np.pi,high=np.pi,nan_policy='omit')/np.pi,linewidth=3,color='r')
    plt.ylabel('FC2-EPG offset (deg)')
    plt.xlabel('FC2 mean activity')
    plt.ylim([0,180])
    plt.yticks(np.arange(0,225,45))
    plt.savefig(os.path.join(savedir,'MeanActivity_vs_FC2_EPG_offset'+'.png'))
    plt.savefig(os.path.join(savedir,'MeanActivity_vs_FC2_EPG_offset'+'.pdf'))
    
    
    plt.figure()
    plt.scatter(pva,180*np.abs(epg)/np.pi,color='k',s=20,alpha=.25)
    pltbins = binspva[1:]-np.diff(binspva)/2
    pltbins = np.tile(pltbins[:,np.newaxis],6)
    plt.plot(pltbins,180*pvamn_epg.T/np.pi,color='r',alpha=.75,linestyle='--')
    plt.plot(pltbins[:,0],180*stats.circmean(pvamn_epg,axis=0,low=-np.pi,high=np.pi,nan_policy='omit')/np.pi,linewidth=3,color='r')
    plt.ylabel('FC2-EPG offset (deg)')
    plt.xlabel('FC2 pva')
    plt.ylim([0,180])
    plt.yticks(np.arange(0,225,45))

    plt.figure()
    plt.scatter(medw,180*np.abs(epg)/np.pi,color='k',s=20,alpha=.25)
    pltbins = binsdf[1:]-np.diff(binsdf)/2
    pltbins = np.tile(pltbins[:,np.newaxis],6)
    plt.plot(pltbins,180*mnmn_epg.T/np.pi,color='r',alpha=.75,linestyle='--')
    plt.plot(pltbins[:,0],180*stats.circmean(mnmn_epg,axis=0,low=-np.pi,high=np.pi,nan_policy='omit')/np.pi,linewidth=3,color='r')
    plt.ylabel('FC2-EPG offset (deg)')
    plt.xlabel('FC2 mean activity')
    plt.ylim([0,180])
    plt.yticks(np.arange(0,225,45))
    
    
    
#%% Model effect of coherence and activity on PFL3 model

from CD_edge_tracking.Models.neuro2pfl3 import pfl3_model

"""
Idea: 
    Run data through PFL3 model, with heading and goal given arbitrary offset of 90 degrees
    bin data by turn amplitude for mean activity vs coherence
"""
mdl = pfl3_model()
binmn = np.arange(0,1.1,.1)
binpva = np.arange(0,.31,.01)
turndata = np.array([])
pvadata = np.array([])
meandata = np.array([])
dchoices = ['FC2','hDeltaC_FC2_EPG']

examplebins = np.array([
    [.3,.04],
    [.3,.13],
    [.3,.19],
    [.5,.04],
    [.5,.13],
    [.5,.19],
    [.5,.25],
    [.8,.04],
    [.8,.13],
    [.8,.19]       ])
example_bumps = np.zeros((len(examplebins),16,13))
cnt = -1
for di,d in enumerate(dchoices):
    data = all_flies[d]
    if di==0:
        ebstr = 'eb'
        fc2str = 'fsb_upper'
    else:
        ebstr = 'eb_ch1'
        fc2str = 'fsb1_ch1'
    
    for f in data:
        cnt = cnt+1
        cxa = data[f].cxa
        ebw = cxa.pdat['wedges_'+ebstr]
        ebp = cxa.pdat['phase_'+ebstr]
        ebw = ug.phase_nulling(ebw,ebp)
        
        fc2 = cxa.pdat['wedges_'+fc2str]
        pva = ug.get_pvas(fc2)
        mn = np.mean(fc2,axis=1)
        fc2p = ug.circ_subtract(cxa.pdat['phase_'+fc2str],np.pi/2) # 90 degree offset
        fc2null = ug.phase_nulling(fc2,cxa.pdat['phase_'+fc2str])
        fc2 = ug.phase_nulling(fc2,fc2p)
        
        L,R,turns = mdl.model_pfl3_all_data(ebw,fc2,goal_weight=10,d7weight=3)
        turndata = np.append(turndata,turns[:])
        meandata = np.append(meandata,mn)
        pvadata = np.append(pvadata,pva)

        for ie,e in enumerate(examplebins):
            dx = (pva<e[1])&(pva>e[1]-.01)&(mn<e[0])&(mn>e[0]-.1)
            fc2ed = fc2null[dx,:]
            if fc2ed.shape[0]>0:
                r = np.random.randint(0,fc2ed.shape[0])
                example_bumps[ie,:,cnt] = fc2ed[r,:]
x = np.arange(0,16)

for i in range(cnt):
    tdata = example_bumps[:,:,i]
    xoffset = 0 
    plt.figure()
    for it,t in enumerate(tdata):
        plt.plot(x+xoffset,t,color='k')
        xoffset +=17
        
    plt.savefig(os.path.join(savedir,'bump_examples_fly_'+str(i) + '.pdf'))
turnmat, xedge, yedge, _ = stats.binned_statistic_2d(
meandata, pvadata, np.abs(turndata),
statistic='mean',   # or 'median' / 'count' depending on what you want to show
bins=[binmn, binpva]
)

tcount, xedge, yedge, _ = stats.binned_statistic_2d(
meandata, pvadata, np.abs(turndata),
statistic='count',   # or 'median' / 'count' depending on what you want to show
bins=[binmn, binpva]
)

turnmat[tcount<50] = np.nan # want more than 5 seconds of data
plt.figure()
plt.imshow(turnmat.T, origin='lower', aspect='auto',
           extent=[binmn[0], binmn[-1], binpva[0], binpva[-1]],
           cmap='viridis')

plt.scatter(examplebins[:,0]-.05,examplebins[:,1]-.005,color='r')
plt.colorbar(label='mean |turn|')
plt.xlabel('FC2 mean activity')
plt.ylabel('FC2 pva')
plt.savefig(os.path.join(savedir,'MeanPVA_vs_PFL_Turn.pdf'))
#%% Plot trajectory snippets with close angular difference
plt.close('all')
dchoices = ['hDeltaC_FC2_EPG']#,'hDeltaC_FC2_EPG_ok_trackers']
minsize = 20
angular_thresh = np.pi*(45/180)
for di,d in enumerate(dchoices):
    data = all_flies[d]
    for f in data:
        plt.figure()
        etp = data[f]
        cxa= etp.cxa
        hdcp = cxa.pdat['phase_fsb2_ch2']
        fc2p = cxa.pdat['phase_fsb1_ch1']
        angdiff = np.abs(ug.circ_subtract(hdcp,fc2p))
        
        # hdcw = cxa.pdat['wedges_fsb2_ch2']
        # fc2w = cxa.pdat['wedges_fsb1_ch1']
        ins = cxa.ft2['instrip']
        ins = np.roll(ins,5)
        x = cxa.ft2['ft_posx'].to_numpy();y = cxa.ft2['ft_posy'].to_numpy();
        x,y = cxa.fictrac_repair(x,y)
        # cvar = ug.rowwise_cov_smth(hdcw,fc2w,10)
        
        
        blocks,bsize = ug.find_blocks(angdiff<=angular_thresh,mergeblocks=True,merg_threshold=3)
        bdx = bsize>=minsize
        blocks = blocks[bdx]
        bsize = bsize[bdx]
        bstr1 = ins[blocks]
        blocks = blocks[bstr1==0]
        bsize = bsize[bstr1==0]
        
        newblocks = np.zeros(len(cvar))
        offset = 0
        for ib,b in enumerate(blocks):
            newblocks[b:b+bsize[ib]] = 1
            tx = x[b:b+bsize[ib]]
            ty = y[b:b+bsize[ib]]
            stx = tx[0]
            sty = ty[0]
            tx = tx-tx[0]
            ty = ty-ty[0]
            bx = x[b-bsize[ib]:b+1]-stx;by = y[b-bsize[ib]:b+1]-sty;
            ax = x[b+bsize[ib]-1:b+bsize[ib]*2]-stx;ay = y[b+bsize[ib]-1:b+bsize[ib]*2]-sty;
            
            print(np.max(tx))
            plt.plot(tx+offset,ty,color='k')
            plt.plot(bx+offset,by,color='r')
            plt.plot(ax+offset,ay,color='b')
            offset += np.max(np.abs(tx))+10
        # plt.figure()
        # plt.plot(cvar>0.015)
        # plt.plot(newblocks,color='g')
        # plt.plot(ins*.2,color='r')
        plt.gca().set_aspect('equal')
#%% plot correlation of hdeltaC and FC2 across trajectories
plt.close('all')
for di,d in enumerate(dchoices):
    data = all_flies[d]
    for f in data:
        etp = data[f]
        cxa = etp.cxa
        hdcw = cxa.pdat['wedges_fsb2_ch2']
        fc2w = cxa.pdat['wedges_fsb1_ch1']
        
        cvar = ug.rowwise_cov_smth(hdcw,fc2w,10)
        etp.cxa.plot_traj_arrow_heat(['fsb2_ch2','fsb1_ch1'],cvar,a_sep=2.5,colormap='coolwarm',cmin=-.015,cmax=.015)
        plt.title(etp.cxa.name)


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
savedir = r'Y:\Data\FCI\FCI_summaries\FC2_hDeltaC_comparison'
for d1,d in enumerate(dchoice):
    plt.figure(figsize=(9,6))
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
        plt.xlabel('hDeltaC ' +  a)
        plt.ylabel('FC2 - hDeltaC offset')
       # plt.ylim([0,150])
        plt.yticks(np.arange(0,160,20))
        #plt.savefig(os.path.join(savedir,'hDeltaC_FC2_phase_align_heuristics.png'))

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
fly =1
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
mns,edges,num = stats.binned_statistic(tdata,tdata2,bins=np.linspace(-np.pi,np.pi,16))
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
#%% Covariance/Pearson R between hDC and FC2 for long returns
data = all_flies['hDeltaC_FC2_EPG']
min_duration = 30*10
outdata = np.zeros((min_duration, len(data),2))
for f in data:
    etp = data[f]
    hdcw = etp.cxa.pdat['wedges_fsb2_ch2']

    fc2w = etp.cxa.pdat['wedges_fsb1_ch1']
    cvs  = ug.rowwise_cov_smth(fc2w,hdcw,10)
    rs = ug.rowwise_pearson_smth(fc2w,hdcw,10)
    jumps = etp.cxa.get_entries_exits_like_jumps()
    jlen = jumps[:,2]-jumps[:,0]
    tjump = jumps[jlen>=min_duration]
    avdata = np.zeros((min_duration,tjump.shape[0],2))
    for ij,j in enumerate(tjump):
        dx = np.arange(j[1],j[1]+min_duration)
        avdata[:,ij,0] = cvs[dx]
        avdata[:,ij,1] =  rs[dx]
    outdata[:,f,:] = np.mean(avdata,axis=1)
    
pltmean = np.mean(outdata,axis=1)  
for r in range(2):
    plt.figure()
    plt.plot(outdata[:,:,r],color='k',alpha=.5)
    plt.plot(pltmean[:,r],color='k',linewidth=2)
#%% Compare covariance accross jumps
data = all_flies['hDeltaC_FC2_EPG']
min_duration = 30*10
outdata = np.zeros((200, len(data),2))

for f in data:
    etp = data[f]
    hdcw = etp.cxa.pdat['wedges_fsb2_ch2']

    fc2w = etp.cxa.pdat['wedges_fsb1_ch1']
    cvs  = ug.rowwise_cov_smth(fc2w,hdcw,10)
    rs = ug.rowwise_pearson_smth(fc2w,hdcw,10)
    jumps = etp.cxa.get_jumps()
    avdata = np.zeros((200,jumps.shape[0],2))
    for ij,j in enumerate(jumps):
        dx = np.arange(j[1],j[1]+min_duration)
        tdata = cvs[dx]
        oldtime = dx-dx[0];newtime = np.linspace(0,oldtime[-1],200)
        avdata[:,ij,0] = np.interp(newtime,oldtime,tdata)
        tdata = rs[dx]
        avdata[:,ij,1] = np.interp(newtime,oldtime,tdata)
       
    outdata[:,f,:] = np.mean(avdata,axis=1)
    
pltmean = np.mean(outdata,axis=1)  
for r in range(2):
    plt.figure()
    plt.plot(outdata[:,:,r],color='k',alpha=.5)
    plt.plot(pltmean[:,r],color='k',linewidth=2)
#%% 


#%% Identify transitions of FC2 phase to hDeltaC phase

plt.close('all')
etp = all_flies['hDeltaC_FC2_EPG'][2]
ins = etp.cxa.ft2['instrip'].to_numpy()
hdc = etp.cxa.pdat['offset_fsb2_ch2_phase'].squeeze()
fc2 = etp.cxa.pdat['offset_fsb1_ch1_phase'].squeeze()
heading = etp.cxa.ft2['ft_heading'].to_numpy()
epgw = etp.cxa.pdat['wedges_eb_ch1']
hdcw = etp.cxa.pdat['wedges_fsb2_ch2']
amphdc = np.mean(etp.cxa.pdat['wedges_fsb2_ch2'],axis=1)
pvahdc = ug.get_pvas(etp.cxa.pdat['wedges_fsb2_ch2'])
fc2w = etp.cxa.pdat['wedges_fsb1_ch1']
t = np.arange(0,len(hdc))/10
cv = ug.rowwise_cov(fc2w,hdcw)
cvs  = ug.rowwise_cov_smth(fc2w,hdcw,10)
r = ug.rowwise_pearson(fc2w,hdcw)
rs = ug.rowwise_pearson_smth(fc2w,hdcw,5)

cve  = ug.rowwise_cov(fc2w,epgw)
cves = ug.rowwise_cov_smth(fc2w,epgw,10)
#plt.plot(t,np.pi+1+cv*50,color='k')
plt.plot(t,t*0+2+np.pi,color='k',linestyle='--')
plt.plot(t,np.pi+2+cvs*50,color=[.7,.7,.7])
#plt.plot(t,np.pi+2+cves*50,color=[0,0,0])
#plt.plot(t[5:],rs,color='m')
#plt.plot(t,r,color='m')
#plt.plot(t,cve*50,color=[.5,.5,.5])
plt.plot(t,t*0,color='r')
colours = uplt.columnar_colours()
plt.scatter(t,fc2,color=colours[2,:],s=2)
plt.plot(t,ins,color='r')
plt.scatter(t,hdc,color=colours[1,:],s=2)
plt.plot(t,heading,color='k')
plt.plot(t,amphdc-np.pi-1.5
         ,color=colours[1,:])
plt.plot(t,pvahdc*5-np.pi-3
         ,color=colours[1,:])
u = ug()
_,_,dd = u.get_velocity(etp.cxa.ft2['ft_posx'].to_numpy(),etp.cxa.ft2['ft_posy'].to_numpy(),t)
plt.plot(t[1:],dd/10-np.pi-2,color=[.5,.5,.5])
all_entries = etp.cxa.get_entries_exits_like_jumps()
for ie,e in enumerate(all_entries):
    plt.text(t[e[0]],2,str(ie))


etp.cxa.plot_traj_arrow_heat(['fsb2_ch2','fsb1_ch1'],cvs,a_sep=2.5,colormap='coolwarm',cmin=-.015,cmax=.015)
#%% Tally of downwind returns
retfly = [[0,8],[0,25],[0,26],[0,27],[0,28],[0,31],[0,48],
          [1,28],[1,41],[1,45],[1,49],
         [2,14],[2,15],[2,23],[2,55], [2,58],[2,70],[2,72],[2,74],[2,79],[2,81],[2,83],[2,84],[2,90],
            [3,3],[3,10],[3,12],[3,17],[3,21],[3,21],[3,27],[3,30],[3,32],[3,37],[3,46],
            [4,2],[4,6],[4,9],[4,13],[4,14],[4,20],[4,28],[4,37],[4,34],
            [5,12],[5,14],[5,15],[5,42]]
#retfly = [[1,28]]
colours = uplt.columnar_colours()
plt.close('all')
from rdp import rdp
for i,r in enumerate(retfly):
    etp = all_flies['hDeltaC_FC2_EPG'][r[0]]
    jumps  = etp.cxa.get_entries_exits_like_jumps()
    tj = jumps[r[1],:]
    tjmin = jumps[r[1]-1,:]
    dx = np.arange(tjmin[1],tjmin[2])
    
    
    x = etp.cxa.ft2['ft_posx'].to_numpy()
    y = etp.cxa.ft2['ft_posy'].to_numpy()
    x,y = etp.cxa.fictrac_repair(x,y)
    
    
    
    hdcw = etp.cxa.pdat['wedges_fsb2_ch2']
    fc2w = etp.cxa.pdat['wedges_fsb1_ch1']
    phase_hdc = etp.cxa.pdat['offset_fsb2_ch2_phase'].to_numpy()[dx]/np.pi
    phase_fc2 = etp.cxa.pdat['offset_fsb1_ch1_phase'].to_numpy()[dx]/np.pi
    phase_fc2o = etp.cxa.pdat['phase_fsb1_ch1'][dx]/np.pi
    phase_hdc2o = etp.cxa.pdat['phase_fsb2_ch2'][dx]/np.pi
    heading  = etp.cxa.ft2['ft_heading'][dx]/np.pi
    cvs = ug.rowwise_cov_smth(hdcw, fc2w, 10)[dx]*10
    phdc = ug.get_pvas(hdcw)[dx]
    pfc2 = ug.get_pvas(fc2w)[dx]
    tx = x[dx];ty= y[dx]
    tx = tx-tx[0];ty = ty-ty[0]
    
    txy = np.append(tx[:,np.newaxis],ty[:,np.newaxis],axis=1)
    msk = rdp(txy,epsilon=20,return_mask=True)
    xyrdp = txy[msk,:]
    
    td  = np.sqrt(np.diff(tx)**2+np.diff(ty)**2)
    tcd = np.append(0,np.cumsum(td))
    t = (dx-dx[0])/10

    plt.figure()
    plt.subplot(1,2,1)
    #plt.plot(tx,ty,color='k')
    #uplt.coloured_line_simple(tx, ty, tcd, 
      #                        'coolwarm', 0, np.max(tcd))
   # uplt.coloured_line(tx, ty, tcd, plt.gca(), cmap='gist_ncar')
    uplt.coloured_line_simple(tx,ty,pfc2,'coolwarm',.05,.15)
    plt.gca().set_aspect('equal')
    plt.scatter(tx[0],ty[0],color='g',marker='*',zorder=5,s=50)
    plt.scatter(tx[-1],ty[-1],color='r',marker='*',zorder=5,s=50)
    #plt.plot(xyrdp[:,0],xyrdp[:,1],color='k')
    cnt = 0
    for t2,i2 in enumerate(tcd):
        if i2>cnt:
            xp = np.array([0,np.sin(phase_fc2[t2]*np.pi)])*3+tx[t2]
            yp = np.array([0,np.cos(phase_fc2[t2]*np.pi)])*3+ty[t2]
            plt.plot(xp,yp,color=colours[2,:])
            
            xp = np.array([0,np.sin(phase_hdc[t2]*np.pi)])*3+tx[t2]
            yp = np.array([0,np.cos(phase_hdc[t2]*np.pi)])*3+ty[t2]
            plt.plot(xp,yp,color=colours[1,:])
            
            cnt =cnt+5
    
    plt.subplot(1,2,2)
    plt.plot(t,cvs,color='k')
    plt.plot(t,pfc2-1.4,color=colours[2,:])
    plt.plot(t,phdc-1.4,color=colours[1,:])
    #uplt.coloured_line(tcd, t*0-1.5, tcd, plt.gca(), cmap='gist_ncar',linewidth=10)
    uplt.coloured_line_simple(t, t*0-1.5,pfc2,'coolwarm',.05,.15,linewidth=10)
    
    
    plt.scatter(t,phase_hdc*.5-.6,color=colours[1,:],s=10)
    plt.scatter(t,phase_fc2*.5-.6,color=colours[2,:],s=10)
    plt.plot(t,heading*.5-.6,color='k',zorder=-1,alpha=.25)
    plt.scatter(t,heading*.5-.6,color='k',zorder=-1,s=5)
    plt.plot([0,t[-1]],[-.6,-.6],color='k')
    plt.plot([0,t[-1]],[-.35,-.35],color='k',linestyle='--')
    plt.plot([0,t[-1]],[-.85,-.85],color='k',linestyle='--')
    plt.xlim([t[0],t[-1]])
    plt.title(r)
    
    
    
    
    # plt.subplot(2,2,3)
    # pmat = np.append(hdcw[dx,:],fc2w[dx,:],axis=1)
    # plt.imshow(np.flipud(pmat.T),cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='none')
    
    # plt.subplot(2,2,4)
    # plt.imshow(np.flipud(pmat.T),cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='none')
    # plt.scatter(t*10,-phase_fc2o*8+8,color='r',s=5)
    # plt.scatter(t*10,-phase_hdc2o*8+8,color=[1,.8,.8],s=5)
    # plt.scatter(t*10,-phase_hdc2o*8+24,color='r',s=5)
    # plt.scatter(t*10,-phase_fc2o*8+24,color=[1,.8,.8],s=5)
#%% Work RDp analysis into above

etp = all_flies['hDeltaC_FC2_EPG'][3]
data = etp.turn2plume_metrics(epsilon=2,regions=regions,jumps='all',plotting='True')
    
#%% von mises estimation of phase changes
testphase = etp.cxa.pdat['phase_fsb2_ch2']
#testphase = etp.cxa.ft2['ft_heading'].to_numpy()
Rs = np.abs(np.mean(np.exp(1j*testphase)))

kappa =  1.0/(2.0* (1.0-Rs))
d,kappas = ug.sliding_von_mises(testphase,100,5,kappa=None)
kappa = np.median(kappas)
d,kappas = ug.sliding_von_mises(testphase,30,5,kappa=kappa)
testplot = testphase[30:-5]
t = np.arange(0,len(testplot))
plt.scatter(t,testplot,color='k',s=5)
plt.scatter(t[d>.4],testplot[d>.4],color='r',s=5)
plt.plot(t,d*np.pi,color='g')

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















    
    
    
    
    
    
    
    
    