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
        
        r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260120\f1\Trial1',
        r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\260120\f1\Trial2',
        
         
         
         
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
datadir =r'Y:\Data\FCI\Hedwig\hDeltaC_68A10_FC2_GCaMP_RCaMP\251202\f1\Trial2'
regions2 = ['fsb1_ch1','fsb2_ch2']
cxa = CX_a(datadir,regions=regions2,yoking=True,denovo=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=True)
#%%
colours = np.array([[81,61,204],[49,99,125],[81,156,205]])/255
phase_fc2 = cxa.pdat['phase_fsb1_ch1']
phase_hdc = cxa.pdat['phase_fsb2_ch2']
ins = cxa.ft2['instrip'].to_numpy()

x = np.arange(0,len(phase_fc2))/10
plt.scatter(x,phase_fc2,color=colours[2,:],s=3)
plt.scatter(x,phase_hdc,color=colours[1,:],s=3)
plt.plot(x,2*ins*np.pi-np.pi,color='r')
amp = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
amp = amp/np.std(amp)
plt.plot(x,amp-8,color=colours[1,:])
amp = np.mean(cxa.pdat['wedges_fsb2_ch1'],axis=1)
amp = amp/np.std(amp)
plt.plot(x,amp-8,color=colours[2,:])
#%% In plume yoking
plt.figure()
jumps = cxa.get_entries_exits_like_jumps(ent_duration=1.5)
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
jumps = cxa.get_entries_exits_like_jumps()
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
#%% EPG bump offset and bias
