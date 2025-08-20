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
from sklearn.decomposition import PCA
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
from matplotlib.collections import PolyCollection
mvthresh = 1 #1mm/s
minsize = 5
plt.close('all')
# Assess phase during returns to jumped plume, look at FC2- EPG phase lag/lead
cmap = plt.get_cmap('coolwarm')
colours = cmap(np.linspace(0, 1, 50))[:,:3]
lscale = np.linspace(-2,2,49)
savedir = r'Y:\Data\FCI\FCI_summaries\FC2_maimon2\BumpAndPhase'
for i in range(len(datadirs)):
    cxa = all_flies[str(i)]
    jumps = cxa.get_jumps()
    
    dx_dt,dy_dt,dd_dt =u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
    stills = dd_dt<mvthresh
    bst,bsz = ug.find_blocks(stills)
    still_start = bst[bsz>=minsize]+1 # add one because the velocity signal is one shorter
    still_size = bsz[bsz>=minsize]
    fsb = cxa.pdat['phase_fsb_upper'] *-cxa.side
    eb = cxa.pdat['phase_eb'] *-cxa.side
    fsb2 = cxa.pdat['phase_fsb_upper']*cxa.side
    fsb_o = cxa.pdat['offset_fsb_upper_phase'].to_numpy() *-cxa.side
    eb_o = cxa.pdat['offset_eb_phase'].to_numpy()*-cxa.side
    eb2 = cxa.pdat['phase_eb'] *cxa.side
    heading = cxa.ft2['ft_heading'].to_numpy()*-cxa.side
    
    ebuw = np.unwrap(eb2)
    huw = np.unwrap(heading)
    
    
    
    stimon = np.where(cxa.ft2['instrip'])[0][0]
    
    
    w_fsb = cxa.pdat['wedges_fsb_upper']
    if cxa.side==-1:
        w_fsb = np.fliplr(w_fsb)
    
    
    wmean = np.mean(w_fsb,axis=1)
    wmeanz = (wmean-np.mean(wmean))/np.std(wmean)
    pdiff = ug.circ_subtract(fsb,eb)
    pva = ug.get_pvas(w_fsb)
    pvaz = (pva-np.mean(pva))/np.std(pva)
    pdiff_vel = ug.circ_vel(pdiff,cxa.pv2['relative_time'],smooth=True,winlength=10)
    fc2_vel = ug.circ_vel(fsb,cxa.pv2['relative_time'],smooth=True,winlength=10)
    #pdiff_vel = ug.circ_vel(pdiff,cxa.pv2['relative_time'],smooth=False)
    pvcorr = ug.time_varying_correlation(pvaz,wmeanz,20)
    for ij,j in enumerate(jumps):
        t_stills = still_start[np.logical_and(still_start>j[1],still_start<j[2])]
        #if len(t_stills)==0:
        fig, ax = plt.subplots(3,1,figsize=(12,9))
        
        
        
        dx = np.arange(j[0],j[2])
        od_off= j[1]-j[0]
        t_fsb = fsb[dx]
        t_h = heading[dx]
        
        
        
        #t_fsb = ug.circ_subtract(t_fsb,t_fsb[od_off])
        t_eb = eb[dx]
        
        h_eb_offset = stats.circmean(ug.circ_subtract(t_h,t_eb))
        t_eb = ug.circ_subtract(t_eb,-h_eb_offset)
        t_fsb = ug.circ_subtract(t_fsb,-h_eb_offset)
        
        #t_eb = ug.circ_subtract(t_eb,t_eb[od_off])
        #t_h = ug.circ_subtract(t_h,t_h[od_off])
        t_vel = dd_dt[dx]
        t_meanz = wmeanz[dx]
        
        
        # headin eb offset
        #c = sg.correlate(t_h,t_eb)
        
        
        
        
        
        verts = []
        x = np.arange(0,len(t_eb),dtype=float)/10
        for ie in range(len(t_eb)-1):
            verts.append([
                (x[ie], t_eb[ie]),
                (x[ie], t_fsb[ie]),
                (x[ie+1], t_fsb[ie+1]),
                (x[ie+1], t_eb[ie+1])
            ]) 
        
        
        for a in range(2):
            if a==0:
                ltmeanz = ug.find_nearest_block(t_meanz,lscale)
                ax[a].set_title('Fly: ' + str(i) + ' Jump: ' +str(ij))
                ax[a].set_ylabel('Phase (deg) / velocity (au)')
            else:
                ltmeanz = ug.find_nearest_block(pvaz[dx],lscale)
                ax[a].set_xlabel('Time (s)')
            poly = PolyCollection(verts ,facecolors=colours[ltmeanz[:-1],:],edgecolors='none') 
            
            ax[a].add_collection(poly)
            ax[a].set_xlim(x.min(), x.max())
            ax[a].set_ylim(min(t_eb.min(), t_fsb.min()), max(t_eb.max(), t_fsb.max()))
            
            ax[a].plot(x,t_fsb,color=[0.2,0.2,1],linestyle='-',linewidth=1)
            ax[a].plot(x,t_eb,color='k',linestyle='-',linewidth=0.5)
            
            
            
            ax[a].plot(x,t_h,color='k',linewidth=2)
            ax[a].plot(x,-5+t_vel/10,color='k')
            
            #plt.plot(pdiff[dx],color='b')
            #plt.plot(pdiff_vel[dx]/2,color='m')
            #plt.plot(fc2_vel[dx]/2,color='m')
            ax[a].plot([x[0],x[-1]],[0,0],color='k',linestyle='--')
            ax[a].plot([x[od_off],x[-1]],[-np.pi/2,-np.pi/2],color='r',linestyle='--')
            ax[a].plot([x[0],x[od_off]],[+np.pi/2,+np.pi/2],color='r',linestyle='--')
            ax[a].set_ylim([-5,np.pi])
            ax[a].fill([x[0],x[od_off],x[od_off],x[0]],[-5,-5,np.pi,np.pi],color=[0.8,0.8,0.8],zorder=-1)
            
            #ax[a].scatter(od_off,1,color='r')
            ax[a].set_yticks([-np.pi,0,np.pi],labels=[-180,0,180])
            
            plt.show()
        tfsb = 7.5*(fsb2[dx]+np.pi)/np.pi
        teb = 7.5*(eb2[dx]+np.pi)/np.pi
        ax[2].imshow(w_fsb[dx,:].T,vmin=0,vmax=1,interpolation='None',aspect='auto')
        ax[2].scatter(x*10,tfsb,color='r')
        ax[2].scatter(x*10,teb,color='k')
        plt.savefig(os.path.join(savedir,'PhaseMeanTopPVAbottom_fly_'+str(i) +'_jump_'+str(ij)+'.png'))
        
#%% better offset determination
plt.close('all')
for i in range(len(datadirs)):
    cxa = all_flies[str(i)]
    eb2 = cxa.pdat['phase_eb'] *-cxa.side
    heading = cxa.ft2['ft_heading'].to_numpy()*-cxa.side
    ebuw = np.unwrap(eb2)
    huw = np.unwrap(heading)
    for ij,j in enumerate(jumps):
        dx = np.arange(j[0],j[2])
    
    
        plt.figure()
        plt.subplot(1,2,1)
        stimon = np.where(cxa.ft2['instrip'])[0][0]
        
        
        plt.plot(ebuw[dx],color='r')
        plt.plot(huw[dx],color='k')
       # cc = sg.correlate(ebuw,huw)
        #ccmx = np.argmax(cc)
        plt.subplot(1,2,2)
        #plt.plot(cc)
        #plt.title(str(len(huw)-ccmx))
        
        x= np.arange(0,len(eb2))
        d = ug.circ_subtract(eb2[dx],heading[dx])
        plt.scatter(x[dx],d,s=5)
        plt.ylim([-np.pi,np.pi])
        #sfilt = ug.savgol_circ(d[dx],winlength=100,polyorder=3)
        #plt.plot(x[1:],sfilt,color='k')
    
    
#%% Phase first 3 seconds after odour offset
plt.close('all')

offset = 0
regions = ['eb','fsb_upper']
for reg in regions:
    plt.figure()
    for i in range(len(datadirs)):
        
        
        cxa = all_flies[str(i)]
        jumps = cxa.get_jumps()
        fsb_o = ug.circ_subtract(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_eb'])*-cxa.side
        
        fsb_o = cxa.pdat['offset_' +reg+'_phase'].to_numpy()*-cxa.side 
        fsb_o = ug.circ_vel(fsb_o,cxa.pv2['relative_time'].to_numpy())
        pcount = 0
        for j in jumps:
            dx = np.arange(j[0],j[2])
            od_off= j[1]-j[0]
            three = od_off+30
            if three<(j[2]-j[0]):
                pcount+=1
                #plt.plot(dx-j[1],eb_o[dx],color='k',alpha=0.3)
                #plt.plot(dx-j[1],fsb_o[dx],color='b',alpha=0.3)
                if pcount==1:
                    pc_array = fsb_o[np.arange(j[1],j[2])]
                    pc_array = pc_array[:30,np.newaxis]
                else:
                    pc_array = np.append(pc_array,fsb_o[np.arange(j[1],j[2])][:30,np.newaxis],axis=1)
                    
        #plt.plot([0,0],[-np.pi,np.pi],color='r')
        #plt.xlim([-50,50])
        pc_array_im = np.cos(pc_array) + 1j*np.sin(pc_array)
        e,ev,xprj = ug.complex_pca(pc_array_im)
        expl = e/np.sum(e)
        scores = np.angle(xprj)
        pcs = np.cumsum(expl)
        top_pcs = np.where(pcs<.95)[0]
        for t in top_pcs:
            x = np.arange(0,30)+offset
            y = fc.unwrap(scores[:,t])
            plt.plot(x,y-10*t,color='b')
            te = np.round(expl[t]*100)
            plt.text(x[0],-10*t,str(te),color='r')
        offset = offset+32
        if i==0:
            grand_pc = pc_array.copy()
        else: 
            grand_pc =np.append(grand_pc,pc_array,axis=1)
    
    grand_pc_im = np.cos(grand_pc) + 1j*np.sin(grand_pc)
    e,ev,xprj = ug.complex_pca(grand_pc_im)
    expl = e/np.sum(e)
    scores = np.angle(xprj)
    pcs = np.cumsum(expl)
    top_pcs = np.where(pcs<.95)[0]
    for t in top_pcs:
        x = np.arange(0,30)+offset
        y = fc.unwrap(scores[:,t])
        plt.plot(x,y-10*t,color='k')
        te = np.round(expl[t]*100)
        plt.text(x[0],-10*t,str(te),color='r')
#%% Covariance matrix in angular velocity space
plt.close('all')
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import leaves_list
regions = ['eb','fsb_upper']
pcount = 0
for i in range(len(datadirs)):
    cxa = all_flies[str(i)]
    jumps = cxa.get_jumps()
    
    for ir,reg in enumerate(regions):
        if ir ==0:
            phase = cxa.pdat['phase_'+reg]*-cxa.side 
        else:
            phase = phase[:,np.newaxis]
            phase = np.append(phase,cxa.pdat['phase_'+reg][:,np.newaxis]*-cxa.side,axis=1 )
            
    #phase = ug.circ_vel(phase,cxa.pv2['relative_time'])
    phasev = np.zeros((len(phase)-1,2))
    for v in range(2):
        phasev[:,v] = ug.circ_vel(phase[:,v],cxa.pv2['relative_time'].to_numpy(),smooth=True,winlength=20)
    phasev = phasev[:,np.newaxis,:]
    phase = phase[:,np.newaxis,:]
    #phase = phasev[:,np.newaxis,:]
    for j in jumps:
            dx = np.arange(j[1],j[2])
            od_off= j[2]-j[1]
            
            if od_off>30:
                pcount+=1

                if pcount==1:
                    grand_pc_an = phase[dx[:30],:,:]
                    
                else:
                    grand_pc_an = np.append(grand_pc_an,phase[dx[:30],:,:],axis=1)
                    
                    
                
                cc = sg.correlate(np.unwrap(phase[dx[:],0,0]),np.unwrap(phase[dx[:],0,1])) # second variable leading first is positive
    #            cc =sg.correlate(phasev[dx[:],0,0],phasev[dx[:],0,1])
                mid = np.argmax(cc)
                print(mid)
                plt.figure()
                plt.subplot(1,2,1)
                plt.plot(cc)
                plt.plot([len(cc)/2,len(cc)/2],[np.min(cc),np.max(cc)],color='r')
                plt.subplot(1,2,2)
                plt.plot(phase[dx[:],0,0],color='k')
                plt.plot(phase[dx[:],0,1],color='b')
                plt.ylim([-np.pi,np.pi])
#%%                
grand_pc_uw = np.unwrap(grand_pc_an[:,:,0],axis=0)        

c = np.corrcoef(grand_pc_uw.T)
# Hierarchical cluster

cluster = AgglomerativeClustering(linkage='single', 
                                compute_distances = True)
cluster.fit(c)
counts = np.zeros(cluster.children_.shape[0])
n_samples = len(cluster.labels_)
for i, merge in enumerate(cluster.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count

linkage_matrix = np.column_stack(
    [cluster.children_, cluster.distances_, counts]
).astype(float)
z = leaves_list(linkage_matrix)
plt.figure()
crank = c[z,:]
crank = crank[:,z]
plt.imshow(crank,vmin=0,vmax=1,interpolation='None')
plt.figure()
x = np.arange(0,30)
for i,iz in  enumerate(z):
    plt.plot(x+i*32,grand_pc_an[:,iz,0],color='k')
    plt.plot(x+i*32,grand_pc_an[:,iz,1],color='b')
        
#%%







for ir,reg in enumerate(regions):
    for i in range(len(datadirs)):
        
        
        cxa = all_flies[str(i)]
        jumps = cxa.get_jumps()
        fsb_o = ug.circ_subtract(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_eb'])*-cxa.side
        
        fsb_o = cxa.pdat['offset_' +reg+'_phase'].to_numpy()*-cxa.side 
        fsb_o_vel = ug.circ_vel(fsb_o,cxa.pv2['relative_time'].to_numpy(),smooth=True)
        pcount = 0
        for j in jumps:
            dx = np.arange(j[0],j[2])
            od_off= j[1]-j[0]
            three = od_off+30
            if three<(j[2]-j[0]):
                pcount+=1

                if pcount==1:
                    pc_array = fsb_o_vel[np.arange(j[1],j[2])]
                    pc_array = pc_array[:30,np.newaxis]
                    
                    pc_array_an = fsb_o[np.arange(j[1],j[2])]
                    pc_array_an = pc_array_an[:30,np.newaxis]
                else:
                    pc_array = np.append(pc_array,fsb_o_vel[np.arange(j[1],j[2])][:30,np.newaxis],axis=1)
                    pc_array_an = np.append(pc_array_an,fsb_o[np.arange(j[1],j[2])][:30,np.newaxis],axis=1)
                    
        if i==0:
            grand_pc = pc_array.copy()
            grand_pc_an = pc_array_an.copy()
        else: 
            grand_pc =np.append(grand_pc,pc_array,axis=1)
            grand_pc_an = np.append(grand_pc_an,pc_array_an,axis=1)
            
grand_pc_uw = np.unwrap(grand_pc_an,axis=0)
c = np.corrcoef(grand_pc_uw.T)
# Hierarchical cluster

cluster = AgglomerativeClustering(linkage='single', 
                                compute_distances = True)
cluster.fit(c)
counts = np.zeros(cluster.children_.shape[0])
n_samples = len(cluster.labels_)
for i, merge in enumerate(cluster.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count

linkage_matrix = np.column_stack(
    [cluster.children_, cluster.distances_, counts]
).astype(float)
z = leaves_list(linkage_matrix)
plt.figure()
crank = c[z,:]
crank = crank[:,z]
plt.imshow(crank,vmin=0,vmax=1,interpolation='None')
plt.figure()
x = np.arange(0,30)
for i,iz in  enumerate(z):
    plt.plot(x+i*32,grand_pc_an[:,iz],color='k')
#%% Phase nulled bump progression for jump returns for individuals
x =  np.arange(0,16)
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
savedir = r'Y:\Data\FCI\FCI_summaries\FC2_maimon2'

bins =5
conds = ['All','walking','still','mid walk']
for c in conds:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plotdata = np.zeros((16,bins*2,2,len(datadirs)))
    for i in range(len(datadirs)):
        
        offset = 0
        cxa = all_flies[str(i)]
        plotdata[:,:,:,i] = cxa.phase_nulled_jump(bins=bins,fsb_names=['eb','fsb_upper'],walk=c)
        
        
    pltmean = np.nanmean(plotdata,axis=3)
    
    cmap = plt.get_cmap('cividis')
    colours = cmap(np.linspace(0,1,bins*2))
    
    asymmetry = np.sum(plotdata[:8,:,:,:]-np.flipud(plotdata[8:,:,:,:]),axis=0)
    
    for i in range(pltmean.shape[1]):
        ax.plot(x+offset,pltmean[:,i,0],color ='k')
        ax.plot(x+offset,plotdata[:,i,0,:],color ='k',alpha=0.2)
        ax.plot(x+offset,pltmean[:,i,1]+1,color ='b')
        ax.plot(x+offset,plotdata[:,i,1,:]+1,color ='b',alpha=0.2)
        if i==bins:
            #plt.plot([offset-0.5,offset-0.5],[0,2],color='r',linestyle='--')
            plt.plot([0,offset-0.5],[0,0],color='r',linewidth=5)
        
        offset = offset+16
    ax.plot(x+offset,pltmean[:,bins-1,1]+1,color =[0.7,0.2,0.7])
    ax.plot(x+offset,plotdata[:,bins-1,1,:]+1,color=[0.7,0.2,0.7],alpha=0.2)
    ax.plot(x+offset,pltmean[:,-1,1]+1,color =[0.0,0.0,1])
    ax.plot(x+offset,plotdata[:,-1,1,:]+1,color=[0.0,0.0,1],alpha=0.2)
    
    ax.plot(x+offset,pltmean[:,bins-1,0],color =[0.7,0.2,0.2])
    ax.plot(x+offset,plotdata[:,bins-1,0,:],color=[0.7,0.2,0.2],alpha=0.2)
    ax.plot(x+offset,pltmean[:,-1,0],color =[0.0,0.0,0])
    ax.plot(x+offset,plotdata[:,-1,0,:],color=[0.0,0.0,0],alpha=0.2)
    
    fig.set_figwidth(14.39)
    plt.title('Phase Nulled bumps: ' + c) 
    plt.ylabel('dF/F0')
    plt.savefig(os.path.join(savedir,'PhaseNulled_'+c+'.pdf'))
    plt.savefig(os.path.join(savedir,'PhaseNulled_'+c+'.png'))
    
    
    plt.figure() # more work on this
    plt.plot([0,bins-1+0.5],[0,0],color='r',linestyle='--')
    plt.plot([bins-1+0.5,bins*2-0.5],[0,0],color='k',linestyle='--')
    for i in range(len(datadirs)):
        plt.scatter(np.arange(0,bins*2),asymmetry[:,0,i],color='k',s=10)
        plt.scatter(np.arange(0,bins*2)+0.25,asymmetry[:,1,i],color='b',s=10)
    plt.ylabel('Bump asymmetry (delta dF/F0)')
    plt.savefig(os.path.join(savedir,'BumpAsymmetry_'+c+'.pdf'))
    plt.savefig(os.path.join(savedir,'BumpAsymmetry_'+c+'.png'))

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
