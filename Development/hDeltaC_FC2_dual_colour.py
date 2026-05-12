# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 17:36:52 2025

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

for i in [1,2]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251017\f1", "Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir,dual_color=True)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
    
    
#%%
datadirs=[ 
    #r"Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251014\f2\Trial1", # not great behaviour proof of principle
          #r"Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251017\f1\Trial1",# 20 entries plume traversals
         # r"Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251017\f1\Trial2",# some entries and exits
         
         # 1020 nm data will be bleedthrough
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251202\f1\Trial1',
        #  r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251202\f1\Trial2', # v nice dataset
        #  r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251202\f1\Trial3',
         
         #1030 nm data, should have less bleedthrough
        #  r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260114\f2\Trial1',
        #  r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260114\f2\Trial2',
        #  r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260114\f2\Trial3',
        # # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260114\f2\Trial4',
        #  r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260114\f2\Trial5'
        
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260120\f1\Trial1',
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260120\f1\Trial2',
        
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial1',
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial2', # made several jumps
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial3' # crossed over plume
         
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial1',
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial2',
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial3',# Lots of jumps
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial4',
         
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260205\f1\Trial1', # Fly walking well then stopping ignoring plumes
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260205\f1\Trial2'
        
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260205\f2\Trial1',#Fly walking well but crossing over plumes a lot.
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260205\f2\Trial2',#Fly walking not tracking
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260205\f2\Trial3',#Fly also not tracking well.
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260205\f2\Trial4',#ACV pulses
        # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260205\f2\Trial5'#ACV pulses 920 nm
        
        r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260211\f1\Trial1',
        r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260211\f1\Trial2'# Excellent tracker
          ]

  

for datadir in datadirs:
   # regions = ['eb','fsb_upper_1','fsb_lower_1','fsb_upper_2']
    regions = ['fsb1','fsb2']
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
    #Channel 2 = Green, Channel 1 = red
    regions = regions = ['fsb1_ch1','fsb1_ch2','fsb2_ch1','fsb2_ch2']
    #regions = ['eb_ch1','eb_ch2','fsb_upper_1_ch1','fsb_lower_1_ch1','fsb_upper_2_ch2']
    #regions = ['fsb_upper_ch1','fsb_upper_ch2','fsb_lower_ch1','fsb_lower_ch2']
    cxa = CX_a(datadir,regions=regions,yoking=False)
    cxa.save_phases()
    try:
        cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
    except:
        print('whoops')
#%% 
regions = ['fsb1','fsb2']

datadir =  r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial3'
regions2 = ['fsb1_ch1','fsb2_ch2']
cxa = CX_a(datadir,regions=regions2,yoking=True,denovo=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=True)
#%%
colours = np.array([[81,61,204],[49,99,125],[81,156,205]])/255
phase_fc2 = cxa.pdat['phase_fsb1_ch1']
heading = cxa.ft2['ft_heading'].to_numpy()
phase_hdc = cxa.pdat['phase_fsb2_ch2'].squeeze()
pva = ug.get_pvas(cxa.pdat['wedges_fsb2_ch1'])
hdcmn = ug.wsumphase(phase_hdc,pva, 10)

ins = cxa.ft2['instrip'].to_numpy()

x = np.arange(0,len(phase_fc2))/10
plt.scatter(x,phase_fc2,color=colours[2,:],s=3)
plt.scatter(x,heading,color='k',s=3)

#plt.plot(x,hdcmn,color=colours[1,:])
plt.scatter(x,hdcmn,color=colours[1,:],s=3)
plt.plot(x,2*ins*np.pi-np.pi,color='r')
amp = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
amp =.5* amp/np.std(amp)
plt.plot(x,amp-8,color=colours[1,:])
amp = np.mean(cxa.pdat['wedges_fsb2_ch1'],axis=1)
amp =.5*amp/np.std(amp)
plt.plot(x,amp-8,color=colours[2,:])

u = ug()
dd,_,_ = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),x)
plt.plot(x[1:],dd/10,color=[.5,.5,.5])

#%% In plume yoking
plt.figure()
jumps = cxa.get_entries_exits_like_jumps(ent_duration=2)
jlen = 30
jdx = (jumps[:,1]-jumps[:,0])>jlen
jumps = jumps[jdx,:]
phase_fc2 = cxa.pdat['phase_fsb1_ch1']
phase_hdc = np.squeeze(cxa.pdat['phase_fsb2_ch2'])
heading = cxa.ft2['ft_heading'].to_numpy()
offset = np.zeros_like(phase_fc2)
last = 0
offmean = np.zeros(len(jumps))
for ij,j in enumerate(jumps):
    dx = np.arange(j[1]-10,j[1])
    theading = stats.circmean(heading[dx],low=-np.pi,high=np.pi)
    tphase = stats.circmean(phase_fc2[dx])
    off = ug.circ_subtract(tphase,theading)
    offset[last:j[1]] = off
    last = j[1]  
    offmean[ij] = off
#phase_fc2 = ug.savgol_circ(phase_fc2,20,3)
#phase_hdc = ug.savgol_circ(phase_hdc,20,3)
#offset = stats.circmean(offmean,low=-np.pi,high=np.pi)
offset_fc2_phase = ug.circ_subtract(phase_fc2,offset)
offset_hdc_phase = ug.circ_subtract(phase_hdc,offset)
cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(offset_fc2_phase)
cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(offset_hdc_phase)
x = np.arange(0,len(phase_fc2))/10
#plt.scatter(x,ug.savgol_circ(offset_fc2_phase,20,3),color=colours[2,:],s=3)
#plt.scatter(x,ug.savgol_circ(offset_hdc_phase,20,3),color=colours[1,:],s=3)
amp = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
amp = amp/np.std(amp)
amp2 = np.mean(cxa.pdat['wedges_fsb1_ch1'],axis=1)
amp2 = amp2/np.std(amp2)


plt.scatter(x,offset_fc2_phase,color=colours[2,:],s=amp2)
plt.scatter(x,offset_hdc_phase,color=colours[1,:],s=amp)
#plt.scatter(x,ug.circ_subtract(phase_hdc,phase_fc2),color=colours[0,:],s=3)
plt.plot(x,heading,color='k',zorder=-1)
plt.plot(x,2*ins*np.pi-np.pi,color='r')

plt.plot(x,amp/2-8,color=colours[1,:])
plt.plot(x,amp2/2-8,color=colours[2,:])
#%% 

#%%
regions2 = ['eb_ch1','eb_ch2','fsb_upper_1_ch1','fsb_upper_2_ch2']
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=True)
regions2 = ['eb_ch1','fsb_upper_1_ch1','fsb_upper_2_ch2']
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=False)

cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=True)

#%%
region1 = "fsb1_ch1"
cxa.plot_traj_arrow(cxa.pdat['offset_'+region1+'_phase'].to_numpy(),np.mean(cxa.pdat['wedges_'+region1]/2,axis=1),a_sep= 2)

region2 = "fsb2_ch2"
cxa.plot_traj_arrow(cxa.pdat['offset_'+region2+'_phase'].to_numpy(),np.mean(cxa.pdat['wedges_'+region2]/2,axis=1),a_sep= 2)

colours = colours = np.array([[49,99,125],[81,156,205]])/255
cxa.plot_traj_arrow_new([region2,region1],a_sep=5,colours =colours)
#%%
cxa.plot_traj_arrow_new(['fsb2_ch2','fsb1_ch1'],a_sep=5)
#%%
x = cxa.ft2['ft_posx'].to_numpy()
jumps = cxa.get_jumps()
jumps = cxa.get_entries_exits_like_jumps()
x = x-x[jumps[0,0]]
jx1 = x[jumps[:,0]]
jx2 = x[jumps[:,1]]
jdiff = jx1-jx2
jdx = np.abs(jdiff)<3 
jumps = jumps[jdx,:]
jmin = 20
jmax = 600
jl = jumps[:,2]-jumps[:,1]
jumps = jumps[np.logical_and(jl<jmax,jl>jmin),:]

outdata = np.zeros((200,2,len(jumps)))
fsb_names = ['fsb1_ch1','fsb2_ch2']
for i,j in enumerate(jumps):
    dx1 = np.arange(j[0],j[1])
    dx2 = np.arange(j[1],j[2])
    
    
    for fi,f in enumerate(fsb_names):
        tphase = cxa.pdat['offset_'+f+'_phase'].to_numpy()[dx1]
        tps = np.sin(tphase)
        tpc = np.cos(tphase)
        new_time = np.linspace(dx1[0],dx1[-1],100) 
        outdata[:100,fi,i] = np.arctan2(np.interp(new_time,dx1,tps),np.interp(new_time,dx1,tpc))
        
        rphase = cxa.pdat['offset_'+f+'_phase'].to_numpy()[dx2]
        rps = np.sin(rphase)
        rpc = np.cos(rphase)
        new_time = np.linspace(dx2[0],dx2[-1],100) 
        outdata[100:,fi,i] = np.arctan2(np.interp(new_time,dx2,rps),np.interp(new_time,dx2,rpc))

out_mean = stats.circmean(outdata,high=np.pi,low=-np.pi,axis=2)

plt.fill([0,100,100,0],[-np.pi,-np.pi,np.pi,np.pi],color=[1,0.4,0.4])
plt.plot(out_mean[:,0],color=colours[2,:])
plt.plot(out_mean[:,1],color=colours[1,:])
#%%
jumps = cxa.get_jumps()
#jumps = cxa.get_jumps()
binsize = .25
binsize = int(binsize*10)
offset=.5
wedges = cxa.pdat['wedges_fsb2_ch2']
wedges_eb = cxa.pdat['wedges_fsb1_ch1']
for ij,j in enumerate(jumps):
    dxe  = np.arange(j[0]-5,j[2])
    
    twed = wedges[dxe,:]
    twede = wedges_eb[dxe,:]
    n_bins = twed.shape[0]//binsize
    onbin = (j[1]-j[0]+5)//binsize
    wed_trimmed =twed[:n_bins*binsize,:]
    wed_binned = wed_trimmed.reshape(n_bins,binsize,16,-1).mean(axis=1)
    
    wed_trimmede =twede[:n_bins*binsize,:]
    wed_binnede = wed_trimmede.reshape(n_bins,binsize,16,-1).mean(axis=1)
    
    
    dxi  = np.arange(j[0],j[1])
    
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(twed)
        # plt.subplot(1,2,2)
        # plt.imshow(wed_binned)
    plt.figure()
    if n_bins>3:
        for w in range(len(wed_binned)):
            tw = wed_binned[w,:]
            twe = wed_binnede[w,:]
            x = np.arange(w*16,(w+1)*16)
            plt.plot(x,twe,color='k')
            plt.plot(x,tw+offset,color='b')
            if w==1:
                plt.plot([x[-1]+.5,x[-1]+.5],[-.1,1.2],color='r')
            if w==onbin:
                plt.plot([x[-1]+.5,x[-1]+.5],[-.1,1.2],color='r')
            if np.mod(w,2)==0:
                plt.fill_between(x,x*0-.1,x*0+1.2,color=[0.8,0.8,0.8],zorder=-1)
                plt.text(x[7],-.2,str((w-1)*binsize/10))
        plt.title(str(ij))
        
#cxa.jump_return_details()
#%% hDeltaC FC2 phase offset jumps
# from plume exit
colours = np.array([
    [81,156,205],
    [0,170,105],
    [80,80,191]])/255
datadirs = [r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial2',
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial3',
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260211\f1\Trial2']
regions2 = ['fsb1_ch1','fsb2_ch2']
plen = 20*10
x = np.arange(0,plen)/10
pltmean = np.zeros((plen,len(datadirs)))*np.nan
pltmeani = np.zeros((plen,2,len(datadirs)))*np.nan
for i,d in enumerate(datadirs):
    cxa = CX_a(d,regions=regions2,yoking=True,denovo=False)
    jumps = cxa.get_jumps(time_threshold=20)
    #jumps = cxa.get_entries_exits_like_jumps()
    fc2 = cxa.pdat['phase_fsb1_ch1']*-cxa.side
    hdc = np.squeeze(cxa.pdat['phase_fsb2_ch2'])*-cxa.side
    offsets = ug.circ_subtract(hdc,fc2)
    plotmat = np.zeros((plen,len(jumps)))*np.nan
    plotmat_indi = np.zeros((plen,2,len(jumps)))*np.nan
    for ij,j in enumerate(jumps):
        dx = np.arange(j[1],j[2])
        if len(dx)>plen:
            dx = dx[:plen]
            
        plotmat[:len(dx),ij] = offsets[dx]
        plotmat_indi[:len(dx),0,ij] = ug.circ_subtract(fc2[dx],fc2[dx][0])
        plotmat_indi[:len(dx),1,ij] = ug.circ_subtract(hdc[dx],fc2[dx][0])
    pltmean[:,i] = stats.circmean(plotmat,low=-np.pi,high=np.pi,axis=1,nan_policy='omit')
    pltmeani[:,:,i] = stats.circmean(plotmat_indi,low=-np.pi,high=np.pi,axis=2,nan_policy='omit')
    
plt.figure(101)
plt.plot(x,pltmean,color='k',alpha=0.5)
plt.plot(x,stats.circmean(pltmean,axis=1,low=-np.pi,high=np.pi,nan_policy='omit'),color='k')

pimean = stats.circmean(pltmeani,axis=2,low=-np.pi,high=np.pi,nan_policy='omit')


plt.figure(102)
plt.plot(pltmeani[:,0,:],x,color=colours[1,:],alpha=0.8,marker='o',linestyle='None',markersize=4)
#plt.plot(x,pimean[:,0],color='b')

plt.plot(pltmeani[:,1,:],x,color=colours[0,:],alpha=0.8,marker='o',linestyle='None',markersize=4)
#plt.plot(x,pimean[:,1],color='r')
plt.plot([0,0],[20,0],color='k',linestyle='--')
plt.plot([-np.pi/2,-np.pi/2],[20,0],color='r',linestyle='--')
plt.xticks(np.arange(-np.pi,np.pi+.1,np.pi/2),labels=np.arange(-180,181,90))
plt.xlabel('Phase (deg)')
plt.xlim([-np.pi,np.pi])
plt.ylabel('Time from plume exit (s)')
plt.ylim([0,15])
plt.figure(101)
plt.ylim([0,np.pi])
    

for i in range(len(datadirs)):
    plt.figure()
    plt.plot(x,pltmeani[:,0,i],color='b',alpha=0.5,marker='o',linestyle='None')

    plt.plot(x,pltmeani[:,1,i],color='r',alpha=0.5,marker='o',linestyle='None')
#%% hDeltaC FC2 phase offset jumps from re-entry
datadirs = [r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial2',
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial3',
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260211\f1\Trial2']
regions2 = ['fsb1_ch1','fsb2_ch2']
plen = 20*10
x = np.arange(0,plen)/10
pltmean = np.zeros((plen,len(datadirs)))*np.nan
pltmeani = np.zeros((plen,2,len(datadirs)))*np.nan
for i,d in enumerate(datadirs):
    cxa = CX_a(d,regions=regions2,yoking=True,denovo=False)
    jumps = cxa.get_jumps(time_threshold=20)
    #jumps = cxa.get_entries_exits_like_jumps()
    fc2 = cxa.pdat['phase_fsb1_ch1']*-cxa.side
    hdc = np.squeeze(cxa.pdat['phase_fsb2_ch2'])*-cxa.side
    offsets = ug.circ_subtract(hdc,fc2)
    plotmat = np.zeros((plen,len(jumps)))*np.nan
    plotmat_indi = np.zeros((plen,2,len(jumps)))*np.nan
    for ij,j in enumerate(jumps):
        dx = np.arange(j[1],j[2])
        offs = fc2[dx[0]]
        if len(dx)>plen:
            dx = dx[-plen:]
            
        plotmat[-len(dx):,ij] = offsets[dx]
        plotmat_indi[-len(dx):,0,ij] = ug.circ_subtract(fc2[dx],offs)
        plotmat_indi[-len(dx):,1,ij] = ug.circ_subtract(hdc[dx],offs)
    pltmean[:,i] = stats.circmean(plotmat,low=-np.pi,high=np.pi,axis=1,nan_policy='omit')
    pltmeani[:,:,i] = stats.circmean(plotmat_indi,low=-np.pi,high=np.pi,axis=2,nan_policy='omit')
    
plt.figure(101)
plt.plot(x,pltmean,color='k',alpha=0.5)
plt.plot(x,stats.circmean(pltmean,axis=1,low=-np.pi,high=np.pi,nan_policy='omit'),color='k')

pimean = stats.circmean(pltmeani,axis=2,low=-np.pi,high=np.pi,nan_policy='omit')


plt.figure(102)
plt.plot(pltmeani[:,0,:],x-20,color=colours[1,:],alpha=0.8,marker='o',linestyle='None',markersize=4)
#plt.plot(x,pimean[:,0],color=colours[1,:])

plt.plot(pltmeani[:,1,:],x-20,color=colours[0,:],alpha=0.8,marker='o',linestyle='None',markersize=4)
#plt.plot(x,pimean[:,1],color=colours[0,:])
plt.plot([0,0],[-20,0],color='k',linestyle='--')
plt.plot([-np.pi/2,-np.pi/2],[-20,0],color='r',linestyle='--')
plt.xticks(np.arange(-np.pi,np.pi+.1,np.pi/2),labels=np.arange(-180,181,90))
plt.xlabel('Phase (deg)')
plt.xlim([-np.pi,np.pi])
plt.ylabel('Time to return (s)')
plt.figure(101)
plt.ylim([0,np.pi])
    

for i in range(len(datadirs)):
    plt.figure()
    plt.plot(x,pltmeani[:,0,i],color='b',alpha=0.5,marker='o',linestyle='None')

    plt.plot(x,pltmeani[:,1,i],color='r',alpha=0.5,marker='o',linestyle='None')
    
    
#%% Recovery dynamics 

colours = np.array([[81,61,204],[49,99,125],[81,156,205]])/255
plt.close('all')
minlen = 50
x = np.arange(0,600)/10
datadirs = [
   r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial2',
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial3',
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260211\f1\Trial2'
   #r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260205\f2\Trial4',
   # r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260205\f2\Trial5'
   ]

regions2 = ['fsb1_ch1','fsb2_ch2']
meandat = np.zeros((600,len(datadirs)))
for di,f in enumerate(datadirs):

    
    cxa = CX_a(f,regions=regions2,yoking=True,denovo=False)
    amp = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
    jumps = cxa.get_entries_exits_like_jumps()
    #jumps = cxa.get_jumps()
    jumplen = jumps[:,2]-jumps[:,1]
    jumps = jumps[jumplen>minlen]
    all_js = np.zeros((600,len(jumps)))
    for i,j in enumerate(jumps):
        dx = np.arange(j[1],j[2])
       # plt.plot(amp[dx],color='k',alpha=0.3)
        tamp = amp[dx]
        amplen  = np.min([600,len(tamp)])
        all_js[:amplen,i] = tamp[:amplen]
        
    all_js[all_js==0] = np.nan    
    meandat[:,di] = np.nanmean(all_js,axis=1)
    
    
    plt.figure(di)
    x = np.arange(0,100)/10
    r = np.arange(0,all_js.shape[1],5)
    colours = uplt.defined_cmap('coolwarm',len(r))

    for i,ir in enumerate(r[:-1]):
        dx = np.arange(ir,r[i+1])
        plt.plot(x,np.nanmean(all_js[:100,dx],axis=1),color=colours[i,:])
plt.figure(101)
x = np.arange(0,600)/10
plt.plot(x[:200],meandat[:200,:],color=colours[0,:],alpha=0.3)
plt.plot(x[:200],np.nanmean(meandat[:200,:],1),color=colours[0,:],alpha=1,linewidth=3)
plt.ylabel('mean dF/F',fontsize=15)
plt.xlabel('time from plume exit (s)',fontsize=15)
plt.xlim([0,20])
plt.xticks(np.arange(0,21,5),fontsize=15)
plt.yticks(np.arange(0,1,.2),fontsize=15)
savedir = 'Y:\Data\FCI\FCI_summaries\hDelta_FC2_Comparison'
#plt.savefig(os.path.join(savedir,'FluorRecoveryAllEntries_68A10.png'))
#%% 
offset = 0
x = np.arange(0,100)/10
r = np.arange(0,all_js.shape[1],5)
colours = uplt.defined_cmap('coolwarm',len(r))

for i,ir in enumerate(r[:-1]):
    dx = np.arange(ir,r[i+1])
    plt.plot(x,np.nanmean(all_js[:100,dx],axis=1),color=colours[i,:])
#%% First entry

colours = np.array([[81,61,204],[49,99,125],[81,156,205]])/255
plt.close('all')
minlen = 50
x = np.arange(0,600)/10
datadirs = [
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial2',
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial3',]

regions2 = ['fsb1_ch1','fsb2_ch2']
meandat = np.zeros((600,len(datadirs)))
for di,f in enumerate(datadirs):

    
    cxa = CX_a(f,regions=regions2,yoking=True,denovo=False)
    amp = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
    jumps = cxa.get_entries_exits_like_jumps()
    #jumps = cxa.get_jumps()
    jumplen = jumps[:,2]-jumps[:,1]
    jumps = jumps[jumplen>minlen]
    all_js = np.zeros((600,len(jumps)))
    j = jumps[0]
    dx = np.arange(j[1],j[2])
    
    tamp = amp[dx]
    amplen  = np.min([600,len(tamp)])
     #all_js[:amplen,i] = tamp[:amplen]
        
    #all_js[all_js==0] = np.nan    
    meandat[:amplen,di] = tamp[:amplen]


plt.plot(x[:200],meandat[:200,:],color=colours[1,:],alpha=0.3)
plt.plot(x[:200],np.nanmean(meandat[:200,:],1),color=colours[1,:],alpha=1,linewidth=3)
plt.ylabel('mean dF/F',fontsize=15)
plt.xlabel('time from plume exit (s)',fontsize=15)
plt.xlim([0,20])
plt.xticks(np.arange(0,21,5),fontsize=15)
plt.yticks(np.arange(0,1,.2),fontsize=15)

#%% Pearson r across track

plt.close('all')
datadirs = [
   r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial2',
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial3',
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260211\f1\Trial2']
colours = np.array([[81,61,204],[49,99,125],[81,156,205]])/255
for d in datadirs:
    cxa = CX_a(d,regions=regions2,yoking=True,denovo=False)
    wfc2 = cxa.pdat['wedges_fsb1_ch1']
    whdc = cxa.pdat['wedges_fsb2_ch2']
    phase_fc2 = cxa.pdat['phase_fsb1_ch1']/np.pi
    phase_hdc = cxa.pdat['phase_fsb2_ch2']/np.pi
    x = np.arange(0,len(phase_fc2))/10
    R = ug.rowwise_pearson(wfc2,whdc)
    plt.figure()
    plt.plot(x,R,color='k')
    ins = cxa.ft2['instrip'].to_numpy()
    plt.plot(x,ins*2-1,color='r')
    
    
    ins = cxa.ft2['instrip'].to_numpy()

    
    plt.scatter(x,phase_fc2,color=colours[2,:],s=3)
    plt.scatter(x,phase_hdc,color=colours[1,:],s=3)

#%% hDeltaC peak finding
for d in datadirs[:2]:
    cxa = CX_a(d,regions=regions2,yoking=True,denovo=False)
    
    ins = cxa.ft2['instrip'].to_numpy()
    inw = np.where(ins>0)[0]
    c,bins = np.histogram(cxa.pdat['phase_fsb2_ch2'],bins=np.linspace(-np.pi,np.pi,200),density=True)
    cs = sg.savgol_filter(c,20,3)
    bins = bins[1:]-np.mean(np.diff(bins))/2
    plt.figure(101)
    plt.plot(bins,cs)
    plt.figure()
    plt.plot(bins,c)
    plt.plot(bins,cs)
    
    pk,_ = sg.find_peaks(cs,prominence=.025)
    if len(pk)<2:
        continue
    pk_phases = bins[pk]
    pk_phase_diff = pk_phases[1]-pk_phases[0]
    # if pk_phase_diff>np.pi:
    upwind_phase = bins[int((pk[0]+pk[1])/2)]
    # else:
    #     upwind_phase = ug.circ_subtract(bins[int((pk[0]+pk[1])/2)],np.pi)
    
    plt.plot([upwind_phase,upwind_phase],[0,.1],color='r')
    
    cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(ug.circ_subtract(cxa.pdat['phase_fsb1_ch1'],upwind_phase))
    cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(ug.circ_subtract(np.squeeze(cxa.pdat['phase_fsb2_ch2']),upwind_phase))
#%%
datadirs = [
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial2',
     r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial3',
     r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260211\f1\Trial2']
plt.close('all')
tdata = np.array([])
regions2 = ['fsb1_ch1','fsb2_ch2']

for di,d in enumerate(datadirs):
    cxa = CX_a(d,regions=regions2,yoking=False,denovo=False)

    jumps = cxa.get_entries_exits_like_jumps()
    jumps = cxa.get_jumps()
    jlen = jumps[:,2]-jumps[:,1]
    jmin = 5
    jumps = jumps[jlen>jmin,:]
    jlen = jlen[jlen>jmin]
    jmax = 200
    jumps = jumps[jlen<jmax,:]
    jlen = jlen[jlen<jmax]
    
    jrank = np.argsort(jlen)
    fc2 = cxa.pdat['phase_fsb1_ch1']
    hdc = np.squeeze(cxa.pdat['phase_fsb2_ch2'])
    pdiff = np.abs(ug.circ_subtract(fc2,hdc))
    tdata  = np.full((600,len(jumps)),np.nan)
    tdata2 = np.full((600,len(jumps)),np.nan)
    for j,ij in enumerate(jrank):
        dx = np.arange(jumps[ij,1],jumps[ij,2])
        x = np.arange(0,len(dx))/10
        plt.figure(100+di)
        plt.plot(x,pdiff[dx],color='k',alpha=.2)
        plt.figure(200+di)
        plt.plot(x-np.max(x),pdiff[dx],color='k',alpha=.2)
        tdata[:len(dx),j] = pdiff[dx] 
        tdata2[-len(dx):,j] = pdiff[dx]
    plt.figure(100+di)
    x = np.arange(0,600)/10
    plt.plot(x,np.nanmedian(tdata,axis=1),color='r')
    
    plt.figure(200+di)
    x = np.arange(-600,0)/10
    plt.plot(x,np.nanmedian(tdata2,axis=1),color='r')    
plt.figure(100+di)
plt.ylim([0,np.pi])
plt.yticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
plt.xlim([0,10])

plt.figure(200+di)
plt.ylim([0,np.pi])
plt.yticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])

plt.xlim([-10,0])
#%% Mutual information
from scipy.ndimage import gaussian_filter
datadir = r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial2'
datadir =  r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial3'
datadir = r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260211\f1\Trial2'

regions2 = ['fsb1_ch1','fsb2_ch2']

cxa = CX_a(datadir,regions=regions2,yoking=True,denovo=False)
from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper
etp = ET_paper(datadir,regions=regions2)
#%% Simple mutual information 
binnum = 20
tbins = 30

datadirs = [
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial2',
     r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial3',
     r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260211\f1\Trial2']
allMI = np.zeros((tbins*2,2,len(datadirs)))
t = np.append(np.linspace(0,1,tbins),np.linspace(1,4,tbins))

for di,d in enumerate(datadirs):
    cxa = CX_a(d,regions=regions2,yoking=True,denovo=False)
    fc2 = cxa.pdat['phase_fsb1_ch1']
    epg = cxa.ft2['ft_heading'].to_numpy() # use EPG as heading - signal offsets do not matter
    hdc = np.squeeze(cxa.pdat['phase_fsb2_ch2'])
    hdc_a = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
    #offset = stats.circmean(cxa.pdat['phase_eb_ch1']-cxa.ft2['ft_heading'].to_numpy())
    
    #epg = ug.circ_subtract(epg,offset)
    #fc2 = ug.circ_subtract(fc2,offset)
    #hdc = ug.circ_subtract(hdc,offset)
    
    etp.cxa.pdat['offset_eb_ch1_phase'] = pd.Series(epg)
    etp.cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(fc2)
    etp.cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(hdc)
    
    cxa.pdat['offset_eb_ch1_phase'] = pd.Series(epg)
    cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(fc2)
    cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(hdc)
    
    # Determine pdensities
    bins = np.linspace(-np.pi,np.pi,binnum)
    pfc2 = np.histogram(fc2,bins,density=True)[0]
    pfc2 = pfc2/np.sum(pfc2)
    
    pepg = np.histogram(epg,bins,density=True)[0]
    pepg = pepg/np.sum(pepg)
    
    phdc = np.histogram(hdc,bins,density=True)[0]
    phdc = phdc/np.sum(phdc)
    # plt.figure()
    # plt.plot(phdc,color='r')
    # plt.plot(pfc2,color='b')
    # plt.plot(pepg,color='k')
    
    pfc2_epg = np.histogram2d(fc2,epg,bins=bins,density=True)[0]
    if binnum>20:
        pfc2_epg = gaussian_filter(pfc2_epg,2)
    pfc2_epg = pfc2_epg/np.sum(pfc2_epg)
    
    pfc2_hdc = np.histogram2d(fc2,hdc,bins=bins,density=True)[0]
    if binnum>20:
        pfc2_hdc = gaussian_filter(pfc2_hdc,2)
    pfc2_hdc = pfc2_hdc/np.sum(pfc2_hdc)
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(pfc2_epg)
    # plt.subplot(1,2,2)
    # plt.imshow(pfc2_hdc)
    
    #p1 is fc2
    #p2 is epg/hdc
    pbins = bins[1:]-np.mean(np.diff(bins))/2
    def MI(p1,p2,p1_p2,bins,p1_obvs,p2_obvs):
        p1_oI = ug.find_nearest_block(p1_obvs,bins) # convert observed phase into indices 
        p2_oI = ug.find_nearest_block(p2_obvs,bins)
        
        MI = 0 
        
        for i in range(len(p1_oI)):
            prob1 = p1[p1_oI[i]]
            prob2 = p2[p2_oI[i]]
            prob1_prob2 = p1_p2[p1_oI[i],p2_oI[i]]
            #MI+= prob2*prob1_prob2*np.log2(prob1_prob2/prob1) # Formula for conditional probabilities
            MI+= prob1_prob2*np.log2((prob1_prob2)/(prob1*prob2))
        return MI
    jumps = cxa.get_jumps()
    # jumps = cxa.get_entries_exits_like_jumps()
    # jlen = jumps[:,2]-jumps[:,1]
    # jmin = 30 
    # jmax = 600
    # jumps = jumps[np.logical_and(jlen>jmin,jlen<jmax),:]
    
    
    
    regions2 = ['fsb1_ch1','eb_ch1','fsb2_ch2']
    cxa.pdat['wedges_eb_ch1'] =cxa.pdat['wedges_fsb1_ch1']
    phase,traj,amp = cxa.pseudo_time_data(cxa.get_jumps(time_threshold=30),bins=tbins,regions=regions2)
    
    for i in range(2):
        if i==0:
            jointp = pfc2_epg
            psecond = pepg
        elif i==1:
            jointp = pfc2_hdc
            psecond = phdc
            
        for ir in range(len(phase)):
            print(i,ir)
            allMI[ir,i,di] = MI(pfc2,psecond,jointp,pbins,phase[ir,0,:],phase[ir,i+1,:])
    
plt.figure()
colours = np.array([[81,61,204],[49,99,125],[81,156,205]])/255
plt.fill([0,1,1,0],[0,0,.1,.1],color=[.8,.8,.8])
plt.plot(t,allMI[:,1,:].squeeze(),color=colours[1,:],alpha=.5)
plt.plot(t,np.mean(allMI[:,1,:],axis=1),color=colours[1,:],alpha=1,linewidth=3)
#plt.plot(t,allMI[:,0,:].squeeze(),color='k')
#%% Mutual information with hdc amplitude

datadirs = [
    r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260131\f1\Trial2',
     r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260201\f1\Trial3',
     r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260211\f1\Trial2']
def entropy(p):
    p = p[p>0]
    return -np.sum(p*np.log2(p))
bin_number = 20
timebins =30
t = np.append(np.linspace(0,1,timebins),np.linspace(1,4,timebins))
allMI2= np.zeros((timebins*2,len(datadirs)))
MIflat = np.zeros((timebins*2,len(datadirs)))
def MI(p1,p2_a,p1_p2_a,bins,binsa,p1_obvs,p2_obvs,a_obvs):
    p1_oI = ug.find_nearest_block(p1_obvs,bins) # convert observed phase into indices 
    p2_oI = ug.find_nearest_block(p2_obvs,bins)
    a_oI = ug.find_nearest_block(a_obvs,binsa)
    
    MI = 0 
    
    for i in range(len(p1_oI)):
        prob1 = p1[p1_oI[i]]
        prob2 = p2_a[p2_oI[i],a_oI[i]]
        prob1_prob2_a = p1_p2_a[p1_oI[i],p2_oI[i],a_oI[i]]
        #MI+= prob2*prob1_prob2*np.log2(prob1_prob2/prob1) # Formula for conditional probabilities
        MI+= prob1_prob2_a*np.log2((prob1_prob2_a)/(prob1*prob2))
    return MI

for di,d in enumerate(datadirs):
    cxa = CX_a(d,regions=regions2,yoking=True,denovo=False)
    bin_number = 20
    fc2 = cxa.pdat['phase_fsb1_ch1']
    #epg = cxa.pdat['phase_eb_ch1']
    hdc = cxa.pdat['phase_fsb2_ch2'].squeeze()
    hdc_a = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
    
    
    #offset = stats.circmean(cxa.pdat['phase_eb_ch1']-cxa.ft2['ft_heading'].to_numpy())
    
    #epg = ug.circ_subtract(epg,offset)
   # fc2 = ug.circ_subtract(fc2,offset)
    #dc = ug.circ_subtract(hdc,offset)
    
    #etp.cxa.pdat['offset_eb_ch1_phase'] = pd.Series(epg)
    etp.cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(fc2)
    etp.cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(hdc)
    
    cxa.pdat['offset_eb_ch1_phase'] = pd.Series(epg)
    cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(fc2)
    cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(hdc)
    
    # Determine pdensities
    bins = np.linspace(-np.pi,np.pi,bin_number)
    pfc2 = np.histogram(fc2,bins,density=True)[0]
    pfc2 = pfc2/np.sum(pfc2)
    
    phdc_amp = np.histogram2d(hdc,hdc_a,bins=bin_number-1,density=True)
    ampbins = phdc_amp[2][1:]
    ampbins = ampbins-np.mean(np.diff(ampbins))/2
    phdc_amp = gaussian_filter(phdc_amp[0],2)
    phdc_amp = phdc_amp/np.sum(phdc_amp)
    
    pfc3_hdc_amp = np.histogramdd( np.concatenate((fc2[:,np.newaxis],hdc[:,np.newaxis],hdc_a[:,np.newaxis]),axis=1),bins=bin_number-1,density=True)[0]
    pfc3_hdc_amp = gaussian_filter(pfc3_hdc_amp,2)
    pfc3_hdc_amp = pfc3_hdc_amp/np.sum(pfc3_hdc_amp)
    # phdc = np.histogram(hdc,bins,density=True)[0]
    # phdc = phdc/np.sum(phdc)
    pfc3_hdc_amp_flat = np.ones_like(pfc3_hdc_amp)*np.mean(pfc3_hdc_amp)
    
    
    #p1 is fc2
    #p2 is epg/hdc
    # a is hdc amplitude
    pbins = bins[1:]-np.mean(np.diff(bins))/2
    
    
    jumps = cxa.get_jumps()
    jumps = cxa.get_entries_exits_like_jumps()
    jlen = jumps[:,2]-jumps[:,1]
    jmin = 30 
    jmax = 600
    jumps = jumps[np.logical_and(jlen>jmin,jlen<jmax),:]
    
    
    
    regions2 = ['fsb1_ch1','fsb2_ch2']
    phase,traj,amp = cxa.pseudo_time_data(jumps,bins=timebins,regions=regions2)

            
    for ir in range(len(phase)):
        
        allMI2[ir,di] = MI(pfc2,phdc_amp,pfc3_hdc_amp,pbins,ampbins,phase[ir,0,:],phase[ir,1,:],amp[ir,1,1,:])/entropy(pfc2)
        MIflat[ir,di] = MI(pfc2,phdc_amp,pfc3_hdc_amp_flat,pbins,ampbins,phase[ir,0,:],phase[ir,1,:],amp[ir,1,1,:])/entropy(pfc2)
    #plt.plot(allMI)
    
plt.figure()
colours = np.array([[81,61,204],[49,99,125],[81,156,205]])/255
plt.fill([0,1,1,0],[0,0,.003,.003],color=[.8,.8,.8])
plt.plot(t,allMI2,color=colours[1,:],alpha=.5)
plt.plot(t,np.mean(allMI2,axis=1),color=colours[1,:],alpha=1,linewidth=3)
