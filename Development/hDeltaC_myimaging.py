# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:37:50 2025

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

for i in [1,2,3,4,5,6]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
#%%
experiment_dirs = [
    #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250313\f1\Trial1",
        #           r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250313\f1\Trial2",
                   
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f1\Trial1",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f1\Trial2",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f1\Trial3",
                   
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial1",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial2",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial3",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial4",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial5",
                  
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250327\f2\Trial3",
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250327\f2\Trial4",
                   
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial1",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial2",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial3",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial4",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial5",
                   
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250411\f1\Trial1",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250411\f1\Trial2",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250411\f1\Trial3",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250411\f1\Trial4",
                   
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial1",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial2",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial3",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial4",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial5",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial6",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial7",
                   
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f2\Trial1",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f2\Trial2",
                   
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial1",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial2",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial6",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial7"
                   
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial1",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial2",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial3",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial4",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial5",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial6"
                   
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial1",#lots of jumps
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial2",# no jumps
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial3",# Plough through plumes
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial4",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial5",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial7",
                   
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial1",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial2",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial3",
                   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial4",
                   
                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial1",
                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial2",
                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial3",
                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial4",
                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial5",
                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial6",
                   
                   ]
regions = ['eb','fsb_upper','fsb_lower']
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

experiment_dirs = [
    #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250313\f1\Trial1",#Lots of plume cross overs
#    r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250313\f1\Trial2",#Lots of plume cross overs
                   
                   #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial1",#Not great behaviour
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial2",# Simple plume no jumps
#                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial3",#Made a few jumps
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial4",#Octanol [?] pulses - neuron is inhibited
                   #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial5"#ACV pulses
                   
                   #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250327\f2\Trial3",
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250327\f2\Trial4", #Octanol pulses files missing near end :( reanalyse
                  
                 # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial1",# Walked until first jump
#                   r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial2",# Multiple plumes 2 jumps
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial3",#No jumps just going straight thru
                   #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial4",'Octanol'
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial5",'ACV
                  
#                    r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250411\f1\Trial1", # Image quality not amazing (edges have higher fluor), but loads of jumps, consistent plume edge pointing
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250411\f1\Trial2",
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250411\f1\Trial3",
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250411\f1\Trial4"
                  
                  
                  #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial1", # One entry but encoding prior entry location
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial2", # No jumps but goal encoding
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial3", # Three jumps, not strong goal encoding, mainly look back activity - high dopamine...?
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial4", # V plume, did left plume, failed on right. Looks like animal had right goal but failed to pay attention to it... very interesting
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial5", # Poor tracker
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial6", # ACV pulses
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial7", # octanol pulses
                  
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f2\Trial1", # one odour pulse
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f2\Trial2" # Three jumps, but very good goal encoding for entire experiment, interesting dynamics with locomotion
                  
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial1", # A few jumps
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial2", # Several jumps: good data, though anisotropy in hDeltaC
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial6", # ACV pulses
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial7" # Octanol pulses
                  
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial1", # Lots of plume cross overs
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial2", # Lots of walking not much edge tracking
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial3", # Little edge tracking, lots of walking
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial4", # Traversing across plumes
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial5", #  oct pulses, imaging artefact in pre air period
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial6", # ACV pulses
                  
                  
                  #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial1",#lots of jumps
                  #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial2",# v strip 1.5 plumes
                  #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial3",# Plough through plumes
                 # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial4", #new bump jump, no jumps
                 # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial5",#ACV pulses
                 # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial7",#OCT pulses
                  
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial1",# No jumps
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial2",# No jumps and plume crosses
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial3",# Beautiful tracker
                  # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial4",# v strip. unsuccessful
                  
                 # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial1",#4 jumps and plume cross
                 # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial2", # 13 jumps, beautiful tracker
                  #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial3", # v strip, does two strips
                 # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial4", # failed v strip. crosses over plume
                  r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial5", # Oct pulses
                  r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial6", # ACV pulses
                  
                    ]
plt.close('all')
for e in experiment_dirs:
    cxa = CX_a(e,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    cxa.simple_raw_plot(plotphase=True,regions = ['fsb_upper','fsb_lower'],yk='eb')
    cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
    
    plt.figure()
    wedges = cxa.pdat['wedges_fsb_upper']
    wamp = np.mean(wedges,axis=1)
    plt.plot(wamp,color='k')
    plt.plot(cxa.ft2['mfc3_stpt']/np.max(cxa.ft2['mfc3_stpt']),color='g')
    plt.plot(cxa.ft2['instrip'],color='r')
    
    try :
        plt.figure()
        cxa.mean_jump_arrows()
        cxa.mean_jump_lines()
    except:
        print('no jumps')
#%% Single data check

datadir = r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial4"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
cxa.simple_raw_plot(plotphase=True,regions = ['fsb_upper','fsb_lower'],yk='eb')
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
if set(['train_heading']).issubset(self.ft2):
plt.figure()
cxa.mean_jump_arrows(fsb_names=['fsb_upper'],ascale=100,jsize=5)
#plt.ylim([-40,40])


    
#%% Load all data up
datadirs = [
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250411\f1\Trial1",# Phase recording is not the best - strong pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial3",# Not many jumps, weak pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f2\Trial2", # Strong pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial2",# Strong pointer, just like FC2
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial1", # Points away ******
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial3", # Strong pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial2",# Strong pointer
                ]

all_flies = {}
for i,datadir in enumerate(datadirs):
    print(datadir)
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    all_flies.update({str(i):cxa})
    cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb')
#%% Tally of good flies
plt.close('all')
jumpsizes = [5,3,3,3,3,3,3,3,3,3]
savedir = r'Y:\Data\FCI\FCI_summaries\hDeltaC'
for di,f in enumerate(all_flies):
    cxa = all_flies[f]
    #plt.figure()
    #cxa.mean_jump_arrows(x_offset=di*40,fsb_names=['fsb_upper'],jsize =jumpsizes[di],ascale=100)
    
    plt.figure(1)
    cxa.mean_jump_arrows_cond(xoffset=di*30,asep=10,colourplot='wmean',cond='None',fsb_names=['fsb_upper'])
    plt.title('All phase')
    plt.figure(2)
    cxa.mean_jump_arrows_cond(xoffset=di*30,asep=10,colourplot='wmean',cond='FSB_uppermean')
    plt.title('High mean activity phase')
plt.figure(1)
plt.savefig(os.path.join(savedir,'AllJumpArrows.pdf'))

#%% Odour inhibition timescale
plt.close('all')
savedir = r'Y:\Data\FCI\FCI_summaries\hDeltaC'
odour_delays = np.zeros((len(all_flies),2))
for di,f in enumerate(all_flies):
    cxa = all_flies[f]
    y = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
    fci = fci_regmodel(y,cxa.ft2,cxa.pv2)
    fci.run(['in odour'])
    odour_delays[di,:] = [fci.delay,fci.r2]
print(np.mean(odour_delays[:,0])/10)
#%% Phase nulled bumps

plt.close('all')
savedir =r'Y:\Data\FCI\FCI_summaries\hDeltaC'
x = np.arange(0,16)
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
        
    #ax.plot(x+offset,pltmean[:,bins-1,0],color ='k')
    #ax.plot(x+offset,plotdata[:,bins-1,0,:],color ='k',alpha=0.2)
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


#%%

cxa.mean_jump_arrows(x_offset=di*40,fsb_names=['fsb_upper'],jsize =jumpsizes[di],ascale=100)

#%%
plt.plot(pvanz,color='k')
plt.plot(wmnz,color='b')
plt.plot(bstrn,color='m')
#plt.plot(pvanz,color='g')
plt.plot(ins,color='r')

plt.figure()
plt.imshow(wedges.T, interpolation='None',aspect='auto',cmap='Blues',vmin=0,vmax=1)
plt.plot(-bstrn-2,color='r')
plt.plot(-wmnz-2,color='k')
plt.plot(-pvaz-2,color=[0.3,0.3,0.3])
plt.plot((-ins*5)-2,color='k')
#%% plot jump returns with detail
from matplotlib.collections import PolyCollection

plt.close('all')
savedir = r'Y:\Data\FCI\FCI_summaries\hDeltaC\PhaseModulationWithJumpReturns'
for i in range(len(datadirs)):
    cxa = all_flies[str(i)]
    cxa.jump_return_details(flynum=i)
    for ij in plt.get_fignums():
        fig = plt.figure(ij)
        fig.savefig(os.path.join(savedir,'PhaseMeanTopPVAbottom_fly_'+str(i) +'_jump_'+str(ij)+'.png'))
    plt.close('all')
#%% Plume cross overs
# First pass, plot each trajectory after a plume cross over with a return, then overlay arrows
plt.close('all')
scalefactor=20
angles = np.linspace(-np.pi,np.pi,16)
pre = np.array([])
post = np.array([])
cmap = plt.get_cmap('hsv')
colours = cmap(np.linspace(0, 1, len(datadirs)))[:,:3]
for i1 in range((len(datadirs))):
    cxa = all_flies[str(i1)]
    x = cxa.ft2['ft_posx'].to_numpy()
    y = cxa.ft2['ft_posy'].to_numpy()
    ins = cxa.ft2['instrip'].to_numpy()
    x,y = cxa.fictrac_repair(x,y)
    jumps = cxa.get_jumps()
    entries,exits = cxa.get_entries_exits()
    
    edx = entries>0#jumps[0,0]
    tentries = entries[edx]
    texits = exits[edx]
    
    x_ent = x[tentries]
    x_ex = x[texits]
    ee_diff = x_ent-x_ex
    cross_overs = np.where(np.abs(ee_diff)>5)[0]
    
    # Plot cross over phases
    eb = cxa.pdat['offset_eb_phase'].to_numpy()
    fsb = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    fsb = cxa.pdat['phase_fsb_upper']
    amp_eb = cxa.pdat['amp_eb']
    amp_fsb = cxa.pdat['amp_fsb_lower']
    
    
    # Get peaks in pva
    wedges = cxa.pdat['wedges_'+'fsb_upper']
    weds = np.sum(wedges*np.sin(angles),axis=1)
    wedc = np.sum(wedges*np.cos(angles),axis=1)
    pva  = np.sqrt(weds**2+wedc**2)
    p0 = np.mean(pva[pva<np.percentile(pva,10)])
    pva = (pva-p0)/p0
    pva = pva/np.max(pva)
    tt = cxa.pv2['relative_time'].to_numpy()
    ins = cxa.ft2['instrip'].to_numpy()
    inson = np.where(np.diff(ins)>0)[0]+1
    insoff = np.where(np.diff(ins)<0)[0]+1
    pvsmooth = sg.savgol_filter(pva,30,3)
    pvstd = np.std(pvsmooth)
    peaks,meta = sg.find_peaks(pvsmooth,prominence=pvstd)
    
    
    for i,c in enumerate(cross_overs):
        
        if c<len(tentries)-1:
            dx = np.arange(tentries[c],tentries[c+1])
        else:
            continue
            #dx = np.arange(texits[c-1],len(x))
            
        tx = x[dx]
        ty = y[dx]
        tins = ins[dx]
        
        ddx = x[tentries[c+1]]-x[texits[c]]
        ret_time = (tentries[c+1]-texits[c])/10
        
        pre_dx = np.arange(tentries[c]-5,tentries[c])
        post_dx = np.arange(tentries[c+1]-5,tentries[c+1])
        pre = np.append(pre,stats.circmean(fsb[pre_dx],high=np.pi,low=-np.pi))
        post = np.append(post,stats.circmean(fsb[post_dx],high=np.pi,low=-np.pi))
        
        
        
        if ret_time<60 and np.abs(ddx)<6:
            plt.figure()
            teb = eb[dx]
            tfsb = fsb[dx]
            tampeb = amp_eb[dx] 
            tampfsb = amp_fsb[dx]
            
            
            # plt.plot(teb,color='k')
            # plt.plot(tfsb,color=[0.2,0.2,1])
            # plt.plot(tins,color='r')
            plt.plot(tx,ty,color='k')
            plt.scatter(tx[tins>0],ty[tins>0],color='r')
            arrowdx = np.arange(0,len(teb),5,dtype=int)
            arrowdx,_ = sg.find_peaks(pvsmooth[dx],prominence=pvstd)
            arrowdx = np.arange(0,len(tx),5,dtype=int)
            for a in arrowdx:
                tp = teb[a]
                xa = tx[a]
                ya = ty[a]
                xa2 = xa+np.sin(tp)*tampeb[a]*scalefactor
                ya2 = ya+np.cos(tp)*tampeb[a]*scalefactor
                #plt.plot([xa,xa2],[ya,ya2],color='k')
                
                tp = tfsb[a]
                xa = tx[a]
                ya = ty[a]
                xa2 = xa+np.sin(tp)*tampfsb[a]*scalefactor
                ya2 = ya+np.cos(tp)*tampfsb[a]*scalefactor
                plt.plot([xa,xa2],[ya,ya2],color=[0.2,0.2,1])
            g=plt.gca()
            g.set_aspect('equal', adjustable='box')
        # plt.title('Fly '+str(i1) +' Cross Over '+str(i))
            plt.figure(101)
            premult = np.sign(pre[-1])
            plt.scatter(pre[-1]*premult,post[-1]*premult,c=colours[i1,:])
plt.xlim([0,np.pi])
plt.ylim([-np.pi,np.pi])
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],linestyle='--',color='k')
#plt.plot([-np.pi,np.pi],[np.pi,-np.pi],linestyle='--',color='k')
plt.plot([-np.pi,0],[0,np.pi],color='r',linestyle='--')
plt.plot([0,np.pi],[-np.pi,0],color='r',linestyle='--')

#%%
plt.close('all')
for i1 in range((len(datadirs))):
    cxa = all_flies[str(i1)]
    #cxa.jump_return_details(flynum=i1,index_type='cross overs')
    cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb')

    #cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
#%%
#%%
plt.close('all')
angles = np.linspace(-np.pi,np.pi,16)
for e in experiment_dirs:
    cxa = CX_a(e,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
    tt = cxa.pv2['relative_time'].to_numpy()
    wedges = cxa.pdat['wedges_fsb_upper']
    wamp = np.mean(wedges,axis=1)
    #plt.plot(wamp)
    
    wednorm = cxa.pdat['wedges_fsb_upper']
    wednorm = wednorm/np.max(wednorm,axis=1)[:,np.newaxis]
    
    weds = np.sum(wedges*np.sin(angles),axis=1)
    wedc = np.sum(wedges*np.cos(angles),axis=1)
    pva  = np.sqrt(weds**2+wedc**2)
    p0 = np.mean(pva[pva<np.percentile(pva,10)])
    pva = (pva-p0)/p0
    
    plt.figure()
    plt.plot(tt,cxa.ft2['instrip']*3,color='r')
    a =pva/np.max(pva)
    a[a<0] = 0
    #plt.plot(tt,pva/np.max(pva))
    plt.plot(tt,cxa.pdat['offset_eb_phase'],color='b',alpha=0.5)
    plt.scatter(tt,cxa.pdat['offset_fsb_upper_phase'].to_numpy(),alpha=a,s=5,color='k')


    wedges = np.append(cxa.pdat['wedges_offset_fsb_upper'],cxa.pdat['wedges_offset_fsb_lower'],axis=1)
    plt.figure()
    plt.imshow(wedges.T,vmin=0,vmax=1,interpolation='None',aspect='auto',cmap='Blues')
    phase = cxa.pdat['offset_eb_phase']
    phase2 = cxa.pdat['offset_fsb_upper_phase']
    phase3 = cxa.pdat['offset_fsb_lower_phase']
    phase2 = 7.5+8*phase2/np.pi
    phase = 7.5+8*phase/np.pi
    x = np.arange(0,len(phase2))
    plt.plot(x,phase,color='k',alpha=0.8)
    plt.plot([0,len(phase2)],[8,8],color='k',linestyle='--')
    plt.scatter(x,phase2,alpha=a,s=10,color='g')
    ins = cxa.ft2['instrip']*16
    plt.plot(x,-.5+ins,color='r')
    plt.scatter(x[ins>0],-.5+ins[ins>0],color='r')
    nm = cxa.ft2['net_motion'].to_numpy()
    plt.plot(x,18+-nm*20,color='k')

    

    
    phase = cxa.pdat['offset_eb_phase']
    phase2 = cxa.pdat['offset_fsb_upper_phase']
    phase3 = cxa.pdat['offset_fsb_lower_phase']
    
    plt.figure()
    plt.scatter(phase3,phase2,c=a,s=1) 
    plt.title('Upper vs Lower')
    plt.figure()
    plt.title('EPG versus Upper')
    plt.scatter(phase,phase2,c=a,s=1) 
    plt.figure()
    plt.title('EPG versus Lower')
    plt.scatter(phase,phase3,c=a,s=1) 
    
# weds = np.sum(wednorm*np.sin(angles),axis=1)
# wedc = np.sum(wednorm*np.cos(angles),axis=1)
# pva_norm  = np.sqrt(weds**2+wedc**2)
# p0 = np.mean(pva_norm[pva_norm<np.percentile(pva_norm,10)])
# pva_norm = (pva_norm-p0)/p0
#plt.plot(pva_norm/np.max(pva_norm))

#plt.figure()
#plt.imshow(wedges,vmin=0,vmax=0.5,interpolation='None',aspect='auto',cmap='Blues')
#%% Actual experiment
datadirs = [r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250313\f1\Trial2",#Lots of plume cross overs
               r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial3",#Made a few jumps
               r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial2",
           r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f2\Trial2",#Three jumps
               ]
angles = np.linspace(-np.pi,np.pi,16)
plt.close('all')
for i, e in enumerate(datadirs):

    #cxa = CX_a(e,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    cxa = datadict['f'+str(i)]
    # Mean fluoresence
    plt.figure()
    wedges = cxa.pdat['wedges_fsb_upper']
    wamp = np.mean(wedges,axis=1)
    plt.plot(wamp,color='k')
    ins = cxa.ft2['instrip']
    plt.plot(-1+ins,color='r')
    fcm = fci_regmodel(wamp,cxa.ft2,cxa.pv2)
    fcm.run(['in odour','angular velocity abs','translational vel','ramp to entry'],partition=False,cirftau=[0.3,0.01])
    plt.plot(fcm.predy,color=[0.2,0.3,0.8])
    plt.title(cxa.name+ ' R2:' +str(np.round(fcm.r2,2)))
    
    cxa.plot_traj_arrow_peaks('fsb_upper')
    
    #plt.plot(pvsmooth)
#%%
datadict = {}
for i,e in enumerate(datadirs):

    datadict['f'+str(i)] = CX_a(e,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    
#%%
plt.close('all')
for i, e in enumerate(datadirs):

    #cxa = CX_a(e,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    
    cxa = datadict['f'+str(i)]
    x = cxa.ft2['ft_posx']
    y = cxa.ft2['ft_posy']
    x,y = cxa.fictrac_repair(x,y)
    wedges = cxa.pdat['wedges_'+'fsb_upper']
    weds = np.sum(wedges*np.sin(angles),axis=1)
    wedc = np.sum(wedges*np.cos(angles),axis=1)
    pva  = np.sqrt(weds**2+wedc**2)
    p0 = np.mean(pva[pva<np.percentile(pva,10)])
    pva = (pva-p0)/p0
    pva = pva/np.max(pva)
    tt = cxa.pv2['relative_time'].to_numpy()
    ins = cxa.ft2['instrip'].to_numpy()
    inson = np.where(np.diff(ins)>0)[0]+1
    insoff = np.where(np.diff(ins)<0)[0]+1
    pvsmooth = sg.savgol_filter(pva,30,3)
    pvstd = np.std(pvsmooth)
    phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    phase2 = cxa.pdat['offset_fsb_lower_phase'].to_numpy()
    phase3 = cxa.pdat['offset_eb_phase'].to_numpy()
    peaks,meta = sg.find_peaks(pvsmooth,prominence=pvstd)
    
    plt.figure()
    plt.scatter(ug.circ_subtract(phase[peaks[:-1]],np.pi),ug.circ_subtract(phase[peaks[1:]],np.pi),s=pvsmooth[peaks[1:]]*100)
    plt.xlabel('peak phase t')
    plt.ylabel('peak phase t+1')
    
    plt.figure()
    heading = cxa.ft2['ft_heading']
    plt.scatter(heading[peaks],phase[peaks],color='k')
    plt.xlabel('heading')
    plt.ylabel('peak phase')
    
    plt.figure()
    heading_sub = ug.circ_subtract(heading,np.pi)
    plt.scatter(heading_sub[peaks],phase[peaks])
    plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
    plt.xlabel('rev heading')
    plt.ylabel('peak phase')
    
    plt.figure()
    plt.scatter(phase2[peaks],phase[peaks])
    plt.xlabel('phase lower')
    plt.ylabel('phase upper')
    plt.plot([-np.pi,0],[0,np.pi],color='k',linestyle='--')
    plt.plot([0,np.pi],[-np.pi,0],color='k',linestyle='--')
    
    plt.figure()
    plt.scatter(phase3[peaks],phase[peaks])
    plt.xlabel('phase epg')
    plt.ylabel('phase upper')
    plt.plot([-np.pi,0],[0,np.pi],color='k',linestyle='--')
    plt.plot([0,np.pi],[-np.pi,0],color='k',linestyle='--')
    
    plt.figure()
    plt.scatter(phase3[peaks],phase2[peaks])
    plt.xlabel('phase epg')
    plt.ylabel('phase lower')
    plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
    
    plt.figure()
    heading_sub = ug.circ_subtract(heading,np.pi)
    
    plt.scatter(heading[peaks],ug.circ_subtract(phase[peaks],np.pi))
    plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
    plt.xlabel('heading')
    plt.ylabel('peak phase')
    plt.yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],labels = [0,90,180,-90,0])
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],labels = [-180,-90,0,90,180])
#%%    
    plt.figure()
    heading_sub = ug.circ_subtract(heading,np.pi)
    plt.scatter(heading_sub[peaks],phase[peaks])
    plt.xlabel('rev heading')
    plt.ylabel('peak phase')
    plt.figure()
    
    io_peaks = peaks[peaks>inson[0]]
    ioff_peaks = peaks[peaks>insoff[0]]
    
    for ip, p in enumerate(io_peaks):
        pp = phase[p]
        insp = np.max(inson[inson<p])
        insphase = phase[insp]
        plt.scatter(insphase,pp,c=pvsmooth[insp])

    plt.figure()
    headingsmooth = ug.circ_subtract(sg.savgol_filter(np.unwrap(heading),20,3),0)
    for ip,p in enumerate(ioff_peaks):
        pp = phase[p]
        insp = np.max(insoff[insoff<p])
        iheading = headingsmooth[insp:p]
        iheading180 = ug.circ_subtract(iheading,np.pi)
        plt.scatter(iheading180[0],pp,color='k')
        plt.xlabel('Heading -180 at exit')
        plt.ylabel('Phase at peak')
        
    plt.figure()
    for ip,p in enumerate(ioff_peaks):
        pp = phase[p]
        px = x[p]
        py  = y[p]
        insp = np.max(insoff[insoff<p])
        ex = x[insp]
        ey = y[insp]
        dx = px-ex
        dy = py-px
        ptan = dy/dx
        pan = np.arctan(ptan)
        pant = ug.circ_subtract(pan,np.pi)
        
        #heading = headingsmooth[insp:p]
        #iheading180 = ug.circ_subtract(iheading,np.pi)
        plt.scatter(pan,pp,color='k')
        plt.scatter(pant,pp,color='r')
        plt.xlabel('angle back to plume')
        plt.ylabel('Phase at peak')
        
        # try:
            
        #     #plt.plot(tt[insp:p],iheading)
        #     plt.plot(tt[insp:p],iheading,color='k')
        #     plt.scatter(tt[insp:p],iheading180,color='b')
        #     plt.plot([tt[insp], tt[p]],[pp,pp],color='r')
        # except:
        #     print('b')
    
        
        
#%% ACV pulses
datadirs = [r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial5",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial5",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial6",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial6",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial6",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial5",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial6"
]
regions = ['fsb_upper','fsb_lower']
savedir = r"Y:\Presentations\2025\04_LabMeeting\hDeltaC"
#plt.close('all')
for e in datadirs:

    cxa = CX_a(e,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    plt.figure()
    for i,r in enumerate(regions):
        plt.subplot(2,1,i+1)
        wedges = cxa.pdat['wedges_'+r]
        wamp = np.mean(wedges,axis=1)
        fcm = fci_regmodel(wamp,cxa.ft2,cxa.pv2)
        plt.plot(fcm.ts,wamp,color='k')
        ins = cxa.ft2['instrip']
        plt.plot(fcm.ts,-1+ins,color='r')
       
        
        
        fcm.run(['in odour','angular velocity abs','translational vel'],partition=False,cirftau=[0.3,0.01])
        rm,lm = fcm.set_up_regressors(['in odour','angular velocity abs','translational vel'])
        plt.plot(fcm.ts_y,fcm.predy,color=[0.2,0.3,0.8])

        plt.plot(fcm.ts,rm[:,1]/np.max(rm[:,1])-2,color=[0.3,0.3,0.3])
        plt.plot(fcm.ts,rm[:,2]/np.max(rm[:,2])-3,color=[0.7,0.7,0.7])
        plt.title(cxa.name+ ' R2:' +str(np.round(fcm.r2,2)))
        plt.ylim([-3.2,1])
        plt.savefig(os.path.join(savedir,'ACV_pulses_' +cxa.name +'.pdf'))
#%% Octanol pulses
datadirs = [r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial4",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial4",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial7",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial7",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial5",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial7",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial5"]
regions = ['fsb_upper','fsb_lower']
savedir = r"Y:\Presentations\2025\04_LabMeeting\hDeltaC"
plt.close('all')
for e in datadirs:
    
    cxa = CX_a(e,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    plt.figure()
    for i,r in enumerate(regions):
        plt.subplot(2,1,i+1)
        wedges = cxa.pdat['wedges_'+r]
        wamp = np.mean(wedges,axis=1)
        fcm = fci_regmodel(wamp,cxa.ft2,cxa.pv2)
        plt.plot(fcm.ts,wamp,color='k')
        ins = cxa.ft2['mfc3_stpt']>0
        plt.plot(fcm.ts,-1+ins,color=[0.4,1,0.4])
        
        fcm.run(['in oct','angular velocity abs','translational vel'],partition=False,cirftau=[0.3,0.001])
        rm,lm = fcm.set_up_regressors(['in odour','angular velocity abs','translational vel'])
        plt.plot(fcm.ts_y,fcm.predy,color=[0.2,0.3,0.8])

        plt.plot(fcm.ts,rm[:,1]/np.max(rm[:,1])-2,color=[0.3,0.3,0.3])
        plt.plot(fcm.ts,rm[:,2]/np.max(rm[:,2])-3,color=[0.7,0.7,0.7])
        plt.title(cxa.name+ ' R2:' +str(np.round(fcm.r2,2)))
        
        plt.ylim([-3.2,1])
        plt.savefig(os.path.join(savedir,'Oct_pulses_' +cxa.name +'.pdf'))
#%% Pulse trajectories
datadirs = [
    
   # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial5",#Crap eb phases
#r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial5",# Crap eb phases
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial6",#ok eb phases
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial6",#Good eb phases
#r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial6",# Crap eb phases
#r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial5",#crap eb phases
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial6" # good eb phases # anticorrelation between phase and heading
]
acv_flies = {}
for i,datadir in enumerate(datadirs):
    print(datadir)
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    acv_flies.update({str(i):cxa})

datadirs = [
    
    #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial4",
#r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial4",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial7",#Crap eb
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial7",#Left side anisotropy
#r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250716\f1\Trial5",
#r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial7",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial5" # Anticorrelation between phase and heading
]
oct_flies = {}
for i,datadir in enumerate(datadirs):
    print(datadir)
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    oct_flies.update({str(i):cxa})
#%%
for f in acv_flies:
    cxa = acv_flies[f]
    cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
    plt.title('ACV fly ' + f)
    cxa = oct_flies[f]
    cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
    plt.title('Oct fly ' + f)
#%%
plt.close('all')
for f in acv_flies:
    cxa = acv_flies[f]
    try:
        cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb')
    except:
        print('nah')
        
    plt.title(f)
    
    
    plt.figure()
    phase = cxa.pdat['phase_fsb_upper']
    heading = cxa.ft2['ft_heading'].to_numpy()
    eb = cxa.pdat['phase_eb']
    ins = cxa.ft2['instrip'].to_numpy()
    tt  =np.arange(0,len(phase))/10
    plt.scatter(tt,phase,color='b',s=10)
    plt.scatter(tt,eb,color=[0.3,0.3,0.3],s=5,alpha=0.5)
    plt.plot(tt,ins,color='r')
    plt.plot(tt,heading,color='k')
    plt.title(f + ' ACV')
    
    
    cxa = oct_flies[f]
    try:
        cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb')
    except:
        print('nah')
        
    plt.title(f + ' OCT')
    plt.figure()
    phase = cxa.pdat['phase_fsb_upper']
    heading = -cxa.ft2['ft_heading'].to_numpy()
    eb = cxa.pdat['phase_eb']
    ins = cxa.ft2['mfc3_stpt'].to_numpy()>0
    tt  =np.arange(0,len(phase))/10
    plt.scatter(tt,phase,color='b',s=10)
    plt.scatter(tt,eb,color=[0.3,0.3,0.3],s=5,alpha=0.5)
    plt.plot(tt,ins,color='r')
    plt.plot(tt,heading,color='k')
    plt.title(f + ' Oct')
    plt.figure()
    plt.scatter(eb,phase,c=tt,s=5)
#%% Example data for lab meeting
plt.close('all')
cxa = datadict['f2']
cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb')
#%% Plot example peaks
wedges = cxa.pdat['wedges_'+'fsb_upper']
weds = np.sum(wedges*np.sin(angles),axis=1)
wedc = np.sum(wedges*np.cos(angles),axis=1)
pva  = np.sqrt(weds**2+wedc**2)
p0 = np.mean(pva[pva<np.percentile(pva,10)])
pva = (pva-p0)/p0
pva = pva/np.max(pva)
tt = cxa.pv2['relative_time'].to_numpy()
ins = cxa.ft2['instrip'].to_numpy()
inson = np.where(np.diff(ins)>0)[0]+1
insoff = np.where(np.diff(ins)<0)[0]+1
pvsmooth = sg.savgol_filter(pva,30,3)
pvstd = np.std(pvsmooth)
peaks,meta = sg.find_peaks(pvsmooth,prominence=pvstd)
#%%
r = 'fsb_upper'
angles = np.linspace(-np.pi,np.pi,16)
sins = np.sin(angles)
coss = np.cos(angles)
wedges = cxa.pdat['wedges_'+r]
x = wedges*sins
y = wedges*coss
xm = np.sum(x,axis=1)
ym = np.sum(y,axis=1)
plt.plot([-1,1],[0,0],color='k',linestyle='--')
plt.plot([0,0],[-1,1],color='k',linestyle='--')
for i in range(16):
    plt.plot([0,x[peaks[10],i]],[0,y[peaks[10],i]],color='k')
plt.plot([0,xm[peaks[10]]],[0,ym[peaks[10]]],color='r')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()
plt.savefig(os.path.join(savedir,'PVA.pdf'))
#%% plot PVA with smoothed
plt.figure(figsize=(10,4))
plt.plot(tt,pva,color='k')
plt.plot(tt,pvsmooth,color=[0.2,0.3,0.8])
plt.plot(tt,0.5*ins-0.6,color='r')
plt.scatter(tt[peaks],pvsmooth[peaks],color='r',s= 30,zorder=10)
plt.xlim([0,600])
plt.savefig(os.path.join(savedir,'Coherence_'+cxa.name+'.pdf'))

plt.figure()
wmn = np.mean(wedges,axis=1)
wmn_smooth = sg.savgol_filter(wmn,30,3)
wstd = np.std(wmn_smooth)
pmean,_ = sg.find_peaks(wmn_smooth,prominence=wstd)
plt.plot(tt,wmn,color='k')
plt.plot(tt,wmn_smooth,color=[0.2,0.3,0.8])
plt.plot(tt,0.5*ins-0.6,color='r')
plt.scatter(tt[pmean],wmn[pmean],color='r',s= 30,zorder=10)
plt.xlim([0,600])
#%%
cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb')
plt.scatter(np.ones(len(peaks))*16,peaks,color='r',zorder=10)
#%% Gather all data
datadirs = [r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250313\f1\Trial2",#Lots of plume cross overs
               r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250320\f2\Trial3",#Made a few jumps
               r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250328\f1\Trial2" ]
for i,e in enumerate(datadirs):
    datadict['f'+str(i)] = CX_a(e,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    
rootdir = 'Y:\\Data\\FCI\\AndyData\\hDeltaC_imaging\\csv'
datadirs = ['20220517_hdc_split_60d05_sytgcamp7f',
 '20220627_hdc_split_Fly1',
 '20220627_hdc_split_Fly2',
 '20220628_HDC_sytjGCaMP7f_Fly1',
 #'20220628_HDC_sytjGCaMP7f_Fly1_45-004', 45 degree plume
 '20220629_HDC_split_sytjGCaMP7f_Fly1',
 '20220629_HDC_split_sytjGCaMP7f_Fly3']
for ir, ddir in enumerate(datadirs):
    datadir = os.path.join(rootdir,ddir,"et")
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    datadict['f'+str(ir+i+1)] = CX_a(datadir,Andy='hDeltaC')    
#%%
plt.close('all')
ticks = [-np.pi,-np.pi/2,0,np.pi/2,np.pi]
labs = [-180,-90,0,90,180]
plt.figure(1)
plt.figure(2)
upper_all = np.array([])
upper_before = np.array([])
peak_beforedx = np.array([])
peak_beforef = np.array([])
for i in datadict:
    cxa = datadict[i]
    wedges = cxa.pdat['wedges_'+'fsb_upper']
    weds = np.sum(wedges*np.sin(angles),axis=1)
    wedc = np.sum(wedges*np.cos(angles),axis=1)
    pva  = np.sqrt(weds**2+wedc**2)
    p0 = np.mean(pva[pva<np.percentile(pva,10)])
    pva = (pva-p0)/p0
    pva = pva/np.max(pva)
    tt = cxa.pv2['relative_time'].to_numpy()
    ins = cxa.ft2['instrip'].to_numpy()
    inson = np.where(np.diff(ins)>0)[0]+1
    insoff = np.where(np.diff(ins)<0)[0]+1
    
    
    pvsmooth = sg.savgol_filter(pva,30,3)
    pvstd = np.std(pvsmooth)
    peaks,meta = sg.find_peaks(pvsmooth,prominence=pvstd)
    preplume_peak = np.array([],dtype='int')
    for i2,it in enumerate(inson[1:]):
        pp = peaks[np.logical_and(peaks<it, peaks>inson[i2-1])]
        if len(pp)>0:
            pre_plumedx = np.max(pp)
            preplume_peak = np.append(preplume_peak,pre_plumedx)
        
    try:
        phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
        phase2 = cxa.pdat['offset_fsb_lower_phase'].to_numpy()
        phase3 = cxa.pdat['offset_eb_phase'].to_numpy()
    except:
        phase = cxa.pdat['offset_fsb_upper_phase']
        phase2 = cxa.pdat['offset_fsb_lower_phase']
        phase3 = cxa.pdat['offset_eb_phase']
    preplume_peak = np.unique(preplume_peak)
    pre_plumedx = np.unique(pre_plumedx)
    peak_beforedx = np.append(peak_beforedx,np.unique(preplume_peak))
    peak_beforef = np.append(peak_beforef,np.unique(preplume_peak)*0+int(i[1]))
    plt.figure(1)
    plt.scatter(phase3[peaks],phase[peaks],color='k',alpha=0.5,s=10)
    plt.scatter(phase3[preplume_peak],phase[preplume_peak],color='r',zorder=10,s=10)
    plt.figure(2)
    plt.scatter(phase2[peaks],phase[peaks],color='k',alpha=0.5,s=10)
    plt.scatter(phase2[preplume_peak],phase[preplume_peak],color='r',zorder=10,s=10)
    
    upper_all = np.append(upper_all,phase[peaks])
    upper_before = np.append(upper_before,phase[np.unique(preplume_peak)])
    
plt.figure(1)
plt.xlabel('EPG phase (deg)')
plt.ylabel('upper phase (deg)')
plt.yticks(ticks,labels=labs)
plt.xticks(ticks,labels=labs)
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
plt.plot([-np.pi,0],[0,np.pi],color='k',linestyle='--')
plt.plot([0,np.pi],[-np.pi,0],color='k',linestyle='--')
plt.ylim([-np.pi,np.pi])
plt.xlim([-np.pi,np.pi])
plt.savefig(os.path.join(savedir,'EPG_upper_phase.pdf'))

plt.figure(2)
plt.xlabel('lower phase (deg)')
plt.ylabel('upper phase (deg)')
plt.yticks(ticks,labels=labs)
plt.xticks(ticks,labels=labs)
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
plt.plot([-np.pi,0],[0,np.pi],color='k',linestyle='--')
plt.plot([0,np.pi],[-np.pi,0],color='k',linestyle='--')
plt.ylim([-np.pi,np.pi])
plt.xlim([-np.pi,np.pi])
plt.savefig(os.path.join(savedir,'Upper_lower_phase.pdf'))

plt.figure(3)
bnum = 16
c,b = np.histogram(upper_all,np.linspace(-np.pi,np.pi,bnum),density=True)
c2,b2= np.histogram(upper_before,np.linspace(-np.pi,np.pi,bnum),density=True)
plotbins = b[1:]-np.mean(np.diff(b))/2
plt.plot(plotbins,c,color='k')
plt.plot(plotbins,c2,color='r')
plt.xticks(ticks,labels=labs)
plt.xlabel('upper phase (deg)')
plt.ylabel('probability')
plt.savefig(os.path.join(savedir,'Phase_probability.pdf'))
#%% Plot segments where the phase is in a cross wind direction
# 1. Plot the in between plume section
# 2. See if oriented towards plume edge
plt.close('all')
for i in datadict:
    plt.figure(figsize=(15,15))
    cxa = datadict[i]
    f = int(i[1])
    ins = cxa.ft2['instrip'].to_numpy()
    x = cxa.ft2['ft_posx']
    y = cxa.ft2['ft_posy']
    x,y = cxa.fictrac_repair(x, y)
    x = x.to_numpy()
    y = y.to_numpy()
    inson = np.where(np.diff(ins)>0)[0]+1
    insoff = np.where(np.diff(ins)<0)[0]+1
    fdx = peak_beforef==f
    tphases = upper_before[fdx]
    tpeaks = peak_beforedx[fdx].astype('int')
    phase = cxa.pdat['offset_fsb_upper_phase']
    offset = 0
    xold = 0
    for it,t in enumerate(tpeaks):
        
        st = np.max(insoff[insoff<t])
        ed = np.min(inson[inson>t])
        pdx = np.arange(st,ed)
        t2 = np.max([t-20,pdx[0]+5])
        #if len(pdx>200):
            #pdx = pdx[-200:]
        tx = x[pdx]
        ty = y[pdx]
        txo = tx[-1]
        tyo = ty[-1]
        tx = tx-tx[-1]
        ty = ty-ty[-1]
        
        offset = offset+xold-np.min(tx)+10
        plt.plot(tx+offset,ty,color='k',linewidth=.75)
        plt.scatter(x[t]-txo+offset,y[t]-tyo,color='r',s=10)
        dx = np.sin(tphases[it])*10
        dy = np.cos(tphases[it])*10
        
        #dx = np.sin(-np.pi/2)*10
        #dy = np.cos(-np.pi/2)*10
        #plt.arrow(x[t]-txo+offset,y[t]-tyo,dx,dy,color='b',linewidth=.75,length_includes_head=True,head_width=1)
        plt.plot([tx[0]+offset,x[t]-txo+offset],[ty[0],y[t]-tyo],color='g')
        plt.plot([tx[0]+offset,x[t2]-txo+offset],[ty[0],y[t2]-tyo],color='g')
        plt.plot([x[t]-txo+offset,x[t]-txo+offset+dx],[y[t]-tyo,y[t]-tyo+dy],color='b')
        
        
        dx = np.sin(phase[t2])*10
        dy = np.cos(phase[t2])*10
        plt.plot([x[t2]-txo+offset,x[t2]-txo+offset+dx],[y[t2]-tyo,y[t2]-tyo+dy],color='b')
        
        plt.plot([tx[-1]+offset,tx[-1]+offset], [0,-50],color='k',linestyle='--'  ,linewidth=.75)
        
        xold = np.max(tx)
        

    g = plt.gca()
    g.set_aspect('equal')
    plt.xlim([0,1200])
    #plt.savefig(os.path.join(savedir,'Eg_returns'+cxa.name+'_.pdf'))
#%% Similar to above but check all peaks for look backs to plume exit
plt.close('all')
plt.figure(101)
for i in datadict:
    
    cxa = datadict[i]
    plt.figure(f+1,figsize=(15,15))
    f = int(i[1])
    ins = cxa.ft2['instrip'].to_numpy()
    x = cxa.ft2['ft_posx']
    y = cxa.ft2['ft_posy']
    x,y = cxa.fictrac_repair(x, y)
    x = x.to_numpy()
    y = y.to_numpy()
    inson = np.where(np.diff(ins)>0)[0]+1
    insoff = np.where(np.diff(ins)<0)[0]+1
    
    wedges = cxa.pdat['wedges_'+'fsb_upper']
    weds = np.sum(wedges*np.sin(angles),axis=1)
    wedc = np.sum(wedges*np.cos(angles),axis=1)
    pva  = np.sqrt(weds**2+wedc**2)
    p0 = np.mean(pva[pva<np.percentile(pva,10)])
    pva = (pva-p0)/p0
    pva = pva/np.max(pva)
    tt = cxa.pv2['relative_time'].to_numpy()
    ins = cxa.ft2['instrip'].to_numpy()
    inson = np.where(np.diff(ins)>0)[0]+1
    insoff = np.where(np.diff(ins)<0)[0]+1
    phase = cxa.pdat['offset_fsb_upper_phase']
    
    pvsmooth = sg.savgol_filter(pva,30,3)
    pvstd = np.std(pvsmooth)
    peaks,meta = sg.find_peaks(pvsmooth,prominence=pvstd)
    
    xold = 0
    for i2,io in enumerate(inson[1:]):
        tpeaks = peaks[np.logical_and(peaks>insoff[i2],peaks<io)]
        if len(tpeaks)==0:
            continue
        xdx = np.arange(insoff[i2],io)
        tx = x[xdx]
        ty = y[xdx]
        xo =tx[0]
        yo = ty[0]
        tx = tx-xo
        ty = ty-yo
        
        offset = offset+xold-np.min(tx)+10
        plt.plot(tx+offset,ty,color='k',zorder=3)
        xold = np.max(tx)
        anvec = np.zeros((len(tpeaks),2))
        for ip2 ,p in enumerate(tpeaks):
            px = x[p]-xo
            py = y[p]-yo
            plt.plot([tx[0]+offset,px+offset],[ty[0],py],color='g',zorder=1)
            plt.plot([tx[-1]+offset,px+offset],[ty[-1],py],color='r',zorder=1)
            pp = phase[p]
            dx = np.sin(pp)*10
            dy = np.cos(pp)*10
            plt.plot([px+offset,px+dx+offset],[py,py+dy],color='b',zorder=4)
            vec_p = np.array([dx,dy])
            vec_p = vec_p/np.sqrt(np.sum(vec_p**2))
            vec_r = np.array([tx[0]-px,ty[0]-py])
            vec_r = vec_r/np.sqrt(np.sum(vec_r**2))
            vec_np = np.array([tx[-1]-px,ty[-1]-py])
            vec_np = vec_np/np.sqrt(np.sum(vec_np**2))
            
            p_r_cos = np.sum(vec_p*vec_r)
            p_np_cos = np.sum(vec_p*vec_np)
            
            plt.figure(101+f)
            plt.scatter(p_r_cos,p_np_cos,color='k',s=10)
            plt.figure(f+1)
           # plt.text(tx[-1]+offset,5,str(np.round(p_r_cos,2)))
           # plt.text(tx[-1]+offset,10,str(np.round(p_np_cos,2)))
            anvec[ip2,0]=p_r_cos
            anvec[ip2,1] = p_np_cos
        
        g = plt.gca()
        g.set_aspect('equal')
        plt.figure(101+f)
        plt.plot(anvec[:,0],anvec[:,1],color='k')
        plt.figure(f+1)
    plt.figure(101+f)
    plt.xlabel('Cos to return')
    plt.ylabel('Cos to goal')    
            
    # g = plt.gca()
    # g.set_aspect('equal')
    #plt.xlim([0,1200])
#%%
import matplotlib.gridspec as gridspec
plt.close('all')
i='f0'
cxa = datadict[i]

f = int(i[1])
ins = cxa.ft2['instrip'].to_numpy()
x = cxa.ft2['ft_posx']
y = cxa.ft2['ft_posy']
x,y = cxa.fictrac_repair(x, y)
x = x.to_numpy()
y = y.to_numpy()
inson = np.where(np.diff(ins)>0)[0]+1
insoff = np.where(np.diff(ins)<0)[0]+1

wedges = cxa.pdat['wedges_'+'fsb_upper']
weds = np.sum(wedges*np.sin(angles),axis=1)
wedc = np.sum(wedges*np.cos(angles),axis=1)
pva  = np.sqrt(weds**2+wedc**2)
p0 = np.mean(pva[pva<np.percentile(pva,10)])
pva = (pva-p0)/p0
pva = pva/np.max(pva)
tt = cxa.pv2['relative_time'].to_numpy()
ins = cxa.ft2['instrip'].to_numpy()
inson = np.where(np.diff(ins)>0)[0]+1
insoff = np.where(np.diff(ins)<0)[0]+1
phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()

pvsmooth = sg.savgol_filter(pva,30,3)
pvstd = np.std(pvsmooth)
peaks,meta = sg.find_peaks(pvsmooth,prominence=pvstd)

xold = 0
for i2,io in enumerate(inson[1:]):
    xdx = np.arange(insoff[i2],io)
    tx = x[xdx]
    ty = y[xdx]
    xo =tx[0]
    yo = ty[0]
    tp = phase[xdx]
    tpvs = pvsmooth[xdx]
    pvec = np.append(np.sin(tp[:,np.newaxis]),np.cos(tp[:,np.newaxis]),axis=1)
    backvec = np.append(tx[0]-tx[:,np.newaxis],ty[0]-ty[:,np.newaxis],axis=1)
    forvec = np.append(tx[-1]-tx[:,np.newaxis],ty[-1]-ty[:,np.newaxis],axis=1)
    backvec = backvec/np.sqrt(np.sum(backvec**2,axis=1))[:,np.newaxis]
    forvec = forvec/np.sqrt(np.sum(forvec**2,axis=1))[:,np.newaxis]
    
    backdot = np.diagonal(np.matmul(backvec,pvec.T))
    fordot = np.diagonal(np.matmul(forvec,pvec.T))
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 3)
    tensample = np.round(np.linspace(0,len(tp)-1,10)).astype('int')
    # Span rows 0 to 3 (exclusive), and columns 0 to 2 (exclusive of 2)
    ax1 = fig.add_subplot(gs[0:3, 0:2])
    c = np.linspace(0,1,len(backdot))
    ax1.plot(backdot,fordot,color='k',zorder=1,alpha=0.5,linewidth=0.5)
    ax1.scatter(backdot[tensample],fordot[tensample],s=pvsmooth[xdx][tensample]*50,c=c[tensample],cmap='coolwarm',zorder=2)
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.xlabel('Cos return')
    plt.ylabel('Cos goal')
    
    ax2 = fig.add_subplot(gs[0:3, 2])
    #uplt.coloured_line(tx-tx[-1],ty-ty[-1],c,ax2,cmap='coolwarm',linewidth=1)
    ax2.plot(tx-tx[-1],ty-ty[-1],color='k',linewidth=0.5,zorder=1)
    ax2.scatter(tx[tensample]-tx[-1],ty[tensample]-ty[-1],s=pvsmooth[xdx][tensample]*50,c=c[tensample],cmap='coolwarm',zorder=2)
    ax2.set_aspect('equal')
    ax2.plot([0,0],[-25,25],color='k',linestyle='--')
    plt.xlim([np.min(tx-tx[-1])-5,np.max(tx-tx[-1])+5])
    
    # plot arrows
    
    uplt.plot_arrows(tx[tensample]-tx[-1],ty[tensample]-ty[-1],tp[tensample],10*tpvs[tensample],color='k',linewidth=1,zorder=10)
    for t in tensample:
        plt.plot([tx[t]-tx[-1],0 ],[ty[t]-ty[-1],0],color='g',alpha=0.25,linewidth=0.5)
        plt.plot([tx[t]-tx[-1],tx[0]-tx[-1] ],[ty[t]-ty[-1],ty[0]-ty[-1]],color='r',alpha=0.25,linewidth=0.5)

#%% 
rm,rt= fcm.set_up_regressors(['oct onset','in oct'],cirftau =[0.3,0.01])
plt.plot(rm)