# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:25:21 2024

@author: dowel
"""

from analysis_funs.regression import fci_regmodel

import numpy as np
import pandas as pd
import analysis_funs.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from scipy import stats
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
from analysis_funs.utilities import funcs as fn

plt.rcParams['pdf.fonttype'] = 42 
#%% Image registraion

for i in [2]:
    datadir =os.path.join("Y:\Data\FCI\Hedwig\FC2_maimon2\\240911\\f2\\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
#%% Basic data processing
experiment_dirs = [
    #"Y:\Data\FCI\Hedwig\FC2_maimon2\\240404\\f1\\Trial3",
                   # "Y:\Data\FCI\Hedwig\FC2_maimon2\\240404\\f1\\Trial4",
                   # "Y:\Data\FCI\Hedwig\FC2_maimon2\\240410\\f1\\Trial4",
                   # "Y:\Data\FCI\Hedwig\FC2_maimon2\\240410\\f1\\Trial5", # Issue with jumps
                   # "Y:\Data\FCI\Hedwig\FC2_maimon2\\240411\\f2\\Trial4", # Jumps not saved
                   # "Y:\Data\FCI\Hedwig\FC2_maimon2\\240411\\f2\\Trial5", # Jumps not saved
                   # "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",#Jump
                   # "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",# Jump
                   # "Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2", # jump
                   #"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial1",# Motion artefact
                   #"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2",
                   "Y:\Data\FCI\Hedwig\FC2_maimon2\\240911\\f2\\Trial1",#No ET but Pb imaged
                   #"Y:\Data\FCI\Hedwig\FC2_maimon2\\240911\\f2\\Trial3" # No ET but Pb imaged.
                   ]

regions = ['fsb_lower','fsb_upper','pb']
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
    cxa = CX_a(datadir,regions=np.flipud(regions))
    
    
        
    
    cxa.save_phases()
#%% Data exploration
plt.close('all')
cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='pb')
#cxa.simple_raw_plot(plotphase=True)
#%%
plt.figure()
plt.scatter(cxa.phase_eb,cxa.phase[:,0],s=2)
plt.scatter(cxa.phase_eb[i_s>0],cxa.phase[i_s>0,0],s=2)
plt.figure()
plt.scatter(cxa.pdat['offset_eb_phase'],cxa.pdat['offset_fsb_upper_phase'],s=2)
plt.scatter(cxa.pdat['offset_eb_phase'][i_s>0],cxa.pdat['offset_fsb_upper_phase'][i_s>0],s=2)
plt.figure()
plt.scatter(cxa.phase_eb,cxa.ft2['ft_heading'],s=2)
plt.scatter(cxa.phase[:,0],cxa.ft2['ft_heading'],s=2)

plt.scatter(cxa.phase[:,0],cxa.phase[:,1],s=1)
# %% Analysis of plume returns
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2"]
datadir =datadirs[3]
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
cxa.von_mises_fit()
# %% Plume transition to return
savedir= 'Y:\\Data\\FCI\\FCI_summaries\\FC2_maimon2'
plt.close('all')
for datadir in datadirs:
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    cxa.von_mises_fit()
    cxa.get_jumps()
    s = cxa.side+1
    s = int(s/2)
    s2 = (cxa.side*-1)+1
    s2 = int(s2/2)
    y = cxa.von_mises['predside_fsb_upper'][:,s]-cxa.von_mises['predside_fsb_upper'][:,s2]
    #plt.figure()
    #plt.plot(y)
    fc = fci_regmodel(y,cxa.ft2,cxa.pv2)
    plt.figure(figsize=(20,20))
    fc.mean_traj_nF_jump(fc.ca,plotjumps=True,cmx=0.4)
    
    plt.savefig(os.path.join(savedir,'von_Mises_Jumps_'+ cxa.name+ '.png'))
    plt.savefig(os.path.join(savedir,'von_Mises_Jumps_' +cxa.name +'.pdf'))
#%% Transition probability analysis
cvarray = np.ones(10)
plt.close('all')
off1 = 0
off2 = 0
bin_edges = np.linspace(-np.pi,np.pi,17)
bump = cxa.pdat['fit_wedges_fsb_upper']
from scipy import ndimage
bumpf = ndimage.gaussian_filter1d(bump,7,axis=0)
dx = cxa.get_jumps()
p = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
t = cxa.pv2['relative_time'].to_numpy()
h = cxa.ft2['ft_heading'].to_numpy()
y = cxa.von_mises['predside_fsb_upper'][:,s]-cxa.von_mises['predside_fsb_upper'][:,s2]

for i,e in enumerate(dx):
    fig,(ax1,ax2) = plt.subplots(2,1)
    tp = p[e[1]:e[2]]
    tt = t[e[1]:e[2]]
    th = h[e[1]:e[2]]
    tb = bumpf[e[1]:e[2],:]
    tpred = y[e[1]:e[2]]
    params = np.zeros((len(tp),3))
    psum = np.zeros(len(tp))
    for m,b in enumerate(tb):
        bm = b-min(b)
        bm = bm/np.sum(bm)
        pdf = stats.rv_histogram((bm, bin_edges))
        samples = pdf.rvs(size=5000)
        params[m,:] = stats.vonmises.fit(samples)
        pcd = stats.vonmises.cdf([-3*np.pi/4,-np.pi/4],params[m,0],params[m,1])
        pcdf = pcd[1]-pcd[0]
        psum[m] = pcdf
 
    tt = tt-np.max(tt)
    ax1.plot(tt,tp-off1,color='b')
    ax1.plot(tt,params[:,1]-off1,color='m')
    ax1.plot(tt,tt*0-np.pi/2-off1,color='k',linestyle='--')
    #ax1.set_ylim([-np.pi,np.pi])
    ax2.plot(tt,psum-off2,color='k')
    ax2.plot(tt,tpred-off2,color='g')
    #off1 = off1+np.pi*2.5 
    #off2 = off2+1.25
    ax2.plot(tt[[0,-1]],[0,0],color='r',linestyle='--')
    ax2.set_xlim([-60,0])
#%%
b = bump[700,:]
b = b-min(b)
bm = np.round(1000*b)
bin_edges = np.linspace(-np.pi,np.pi,17)
pdf = stats.rv_histogram((bm, bin_edges))
samples = pdf.rvs(size=100)
params = stats.vonmises.fit(samples)
mn = 8*params[1]/np.pi
plt.plot(bm)
plt.scatter(mn+8,0.1)
# %% Analysis of jumps
savedir = "Y:\\Data\\FCI\\FCI_summaries\\FC2_maimon2"
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2"]
plt.close('all')
x_offset = 0

for datadir in datadirs:
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    cxa.save_phases()
    cxa.mean_jump_arrows(x_offset)
    cxa.mean_jump_lines()
    x_offset = x_offset+30
    plt.savefig(os.path.join(savedir,'EgJumps' + name +'.png'))
    plt.savefig(os.path.join(savedir,'EgJumps'+ name+ '.pdf'))


#plt.ylim([-50,30])
savedir= 'Y:\\Data\\FCI\\FCI_summaries\\FC2_maimon2'
#plt.savefig(os.path.join(savedir,'MeanJumps.png'))
#plt.savefig(os.path.join(savedir,'MeanJumps.pdf'))

#%% Meno Et comp

for i,datadir  in enumerate(datadirs):
    print(i)
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    meno_array,et_array,et_array_ex,pltbins = cxa.plume_meno_comp()
    meno_array = np.expand_dims(meno_array,2)
    et_array = np.expand_dims(et_array,2)
    et_array_ex = np.expand_dims(et_array_ex,2)
    if i==0:
        plt_meno = meno_array
        plt_et = et_array
        plt_et_ex = et_array_ex
    else:
        plt_meno = np.append(plt_meno,meno_array,axis=2)
        plt_et = np.append(plt_et,et_array,axis=2)
        plt_et_ex = np.append(plt_et_ex,et_array,axis=2)
    
#%% Regression analysis of bump amplitude
regchoice = ['odour onset', 'odour offset', 'in odour', 
                             'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
  'angular velocity pos','angular velocity neg','translational vel','ramp down since exit','ramp to entry']

dr2mat = np.zeros((len(datadirs),len(regchoice)))
dr2mat_max = np.zeros((len(datadirs),len(regchoice)))
angles = np.linspace(-np.pi,np.pi,16)
savedir = "Y:\\Data\\FCI\\FCI_summaries\\FC2_maimon2"
for ir, datadir in enumerate(datadirs):
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    
    #y = cxa.pdat['amp_fsb_upper']
    weds = np.sum(cxa.pdat['fit_wedges_fsb_upper']*np.sin(angles),axis=1)
    wedc = np.sum(cxa.pdat['fit_wedges_fsb_upper']*np.cos(angles),axis=1)
    y  = np.sqrt(weds**2+wedc**2)
    
    ft2 = cxa.ft2
    pv2 = cxa.pv2
    fc = fci_regmodel(y,ft2,pv2)
    fc.rebaseline(500)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    dr2mat[ir,:] = (-fc.dR2_mean)*np.sign(fc.coeff_cv[:-1])
    
    fc.plot_example_flur()
    plt.title('Fly: ' + str(ir) +  ' R2:' +str(fc.cvR2))
    plt.savefig(os.path.join(savedir,'EgFit_' + str(ir)+ '.png'))
    plt.figure()
    plt.title(str(ir))
    
    plt.savefig(os.path.join(savedir,'PVA_withreg_' + str(ir)+ '.png'))
    
    y = np.mean(cxa.pdat['wedges_offset_fsb_upper'],axis=1)
    ft2 = cxa.ft2
    pv2 = cxa.pv2
    fc = fci_regmodel(y,ft2,pv2)
    fc.rebaseline(500)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    dr2mat_max[ir,:] = (-fc.dR2_mean)*np.sign(fc.coeff_cv[:-1])
    
    fc.plot_example_flur()
    plt.title('Fly: ' + str(ir) +  ' R2:' +str(fc.cvR2))
    plt.savefig(os.path.join(savedir,'EgFit_' + str(ir)+ '.png'))
    plt.figure()
    plt.title(str(ir))
    plt.savefig(os.path.join(savedir,'Ca_withreg_' + str(ir)+ '.png'))

#%% 
plt.figure()
x = np.arange(0,len(regchoice))
plt.plot([0,len(regchoice)],[0,0],linestyle='--',color='k')
plt.plot(x,np.transpose(dr2mat),color='k',alpha=0.3)
plt.plot(x,np.mean(dr2mat,axis=0),color=[0.2,0.2,1],linewidth=2)

plt.xticks(x,labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.title('PVA amplitude regression')
plt.ylabel('Signed dR2')
plt.savefig(os.path.join(savedir,'Reg_Bump_PVA_dR2.png'))
plt.savefig(os.path.join(savedir,'Reg_Bump_PVA_dR2.pdf'))
plt.figure()
x = np.arange(0,len(regchoice))
plt.plot([0,len(regchoice)],[0,0],linestyle='--',color='k')
plt.plot(x,np.transpose(dr2mat_max),color='k',alpha=0.3)
plt.plot(x,np.mean(dr2mat_max,axis=0),color=[0.2,0.2,1],linewidth=2)
plt.xticks(x,labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.title('Mean columns')
plt.ylabel('Signed dR2')
plt.savefig(os.path.join(savedir,'Reg_Bump_Mean_dR2.png'))
plt.savefig(os.path.join(savedir,'Reg_Bump_Max_dR2.pdf'))
#%% Mean bump amplitudes
plotmat = np.zeros((100,len(datadirs)))
for ir, datadir in enumerate(datadirs):
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)

    y = np.mean(cxa.pdat['wedges_offset_fsb_upper'],axis=1)
    ft2 = cxa.ft2
    pv2 = cxa.pv2
    fc = fci_regmodel(y,ft2,pv2)
    t,yt = fc.mean_traj_nF(use_rebase=True)
    plotmat[:,ir] = yt
   
#%%
mi = -0.04
mxi = 0.04
plotmat2 = plotmat-np.mean(plotmat[1:49,:],axis=0)
pltm = np.mean(plotmat2,axis=1)
plt.plot(plotmat2,color=[0.2,0.2,1],alpha=0.3)
plt.plot(pltm,color=[0.2,0.2,1],linewidth=2)
plt.plot([49,49],[mi,mxi],color='k',linestyle='--')

plt.fill([49,99,99,49],[mi,mi,mxi,mxi],color=[0.7,0.7,0.7])
plt.plot([0,99],[0,0],color='k',linestyle='--')
plt.plot([49,49],[mi,mxi],color='k',linestyle='--')
plt.plot([0,0],[mi,mxi],color='k',linestyle='--')
plt.plot([99,99],[mi,mxi],color='k',linestyle='--')

plt.xticks([49,99],labels=['Plume entry','Plume exit'])
plt.xlim([0,101])
plt.ylabel('Mean norm fluor')
plt.savefig(os.path.join(savedir,'MeanFluorMod.png'))
#%% Predict turn direction from offset between FSB and EB heading signals
from scipy.ndimage import gaussian_filter
phase = cxa.phase[:,0]
phase_eb = cxa.phase_eb
sindiff = np.sin(phase)-np.sin(phase_eb)
cosdiff = np.cos(phase)-np.cos(phase_eb)
d_phase = np.arctan2(sindiff,cosdiff)
d_phase = phase-phase_eb
d_phasen = np.arctan2(np.sin(d_phase),np.cos(d_phase))
#plt.plot(phase,color=[0.2,0.2,1])
#plt.plot(phase_eb,color='k')
times = cxa.pv2['relative_time']
heading = cxa.ft2['ft_heading']
dheading = np.diff(heading)/np.diff(times)
dheading = np.append([0],dheading)
h99 = np.percentile(np.abs(dheading),99)
dheading[np.abs(dheading)>h99] = h99
#plt.plot(dheading/np.max(dheading))
hf = gaussian_filter(dheading,5)

plt.plot(hf/np.max(hf))
dpf = gaussian_filter(d_phasen,5)
plt.plot(dpf/np.pi,color=[1,0.2,0.2])
plt.plot(cxa.ft2['instrip']*2-1,color='k')
plt.plot([0,len(dpf)],[0,0],color='k',linestyle='--')
#%% 
phase = cxa.pdat['offset_fsb_upper_phase']
plt.plot(phase/np.pi,color=[0.2,0.2,1])
plt.plot(cxa.ft2['ft_heading']/np.pi,color='k')
plt.plot(cxa.ft2['instrip']*2-1,color=[0.7,0,0])
plt.plot([0,len(dpf)],[0,0],color='k',linestyle='--')
#%% Jump phase line plots, to get te

#%% Stationary replay event



        
#%%    
plt.close('all')
savedir = "Y:\\Presentations\\ForVanessa"
colours = np.array([[0,0,0],[0.3,0.3,1],[1,0.3,1]])

fig, ax1 = plt.subplots()
i = 0
tdat = plt_et[:,i,:]
pltmean = np.mean(tdat,axis=1)
pltse  = np.std(tdat,axis=1)/np.sqrt(len(datadirs))
ax1.fill_between(pltbins,pltmean-pltse,pltmean+pltse,color=colours[i,:],alpha=0.5)
ax1.plot(pltbins,pltmean,color=colours[i,:],alpha=1)
ax1.plot([0,0],[0,max(pltmean+pltse)],linestyle='--',color='k',zorder=3)
ax1.set_xlabel('Phase-heading offset (deg)')
ax1.set_xticks([-180,-90,0,90,180])
ax1.set_ylabel('Probability - EB')
ax1.set_title('Plume jump returns')
ax1.set_ylim([0,0.4])
ax2 = ax1.twinx()
ax2.set_ylabel('Probability - FSB')
for i in range(2):
    #plt.plot(pltbins,plt_et[:,i,:],color=colours[i,:],alpha=0.5)
    tdat = plt_et[:,i+1,:]
    pltmean = np.mean(tdat,axis=1)
    pltse  = np.std(tdat,axis=1)/np.sqrt(len(datadirs))
    ax2.fill_between(pltbins,pltmean-pltse,pltmean+pltse,color=colours[i+1,:],alpha=0.5)
    ax2.plot(pltbins,pltmean,color=colours[i+1,:],alpha=1)
    #
fig.tight_layout()  
plt.show()
ax2.set_ylim([0,0.25])
plt.savefig(os.path.join(savedir,'PlumeJumpReturnHisto.pdf'))

fig, ax1 = plt.subplots()
i = 0
tdat = plt_et_ex[:,i,:]
pltmean = np.mean(tdat,axis=1)
pltse  = np.std(tdat,axis=1)/np.sqrt(len(datadirs))
ax1.fill_between(pltbins,pltmean-pltse,pltmean+pltse,color=colours[i,:],alpha=0.5)
ax1.plot(pltbins,pltmean,color=colours[i,:],alpha=1)
ax1.plot([0,0],[0,max(pltmean+pltse)],linestyle='--',color='k',zorder=3)
ax1.set_xlabel('Phase-heading offset (deg)')
ax1.set_xticks([-180,-90,0,90,180])
ax1.set_ylabel('Probability - EB')
ax1.set_title('Plume jump exits')
ax1.set_ylim([0,0.4])
ax2 = ax1.twinx()
ax2.set_ylabel('Probability - FSB')
for i in range(2):
    #plt.plot(pltbins,plt_et[:,i,:],color=colours[i,:],alpha=0.5)
    tdat = plt_et_ex[:,i+1,:]
    pltmean = np.mean(tdat,axis=1)
    pltse  = np.std(tdat,axis=1)/np.sqrt(len(datadirs))
    ax2.fill_between(pltbins,pltmean-pltse,pltmean+pltse,color=colours[i+1,:],alpha=0.5)
    ax2.plot(pltbins,pltmean,color=colours[i+1,:],alpha=1)
    #
ax2.set_ylim([0,0.25])
fig.tight_layout()  
plt.show()
plt.savefig(os.path.join(savedir,'PlumeJumpExitHisto.pdf'))

fig, ax1 = plt.subplots()
i = 0
tdat = plt_meno[:,i,:]
pltmean = np.mean(tdat,axis=1)
pltse  = np.std(tdat,axis=1)/np.sqrt(len(datadirs))
ax1.fill_between(pltbins,pltmean-pltse,pltmean+pltse,color=colours[i,:],alpha=0.5)
ax1.plot(pltbins,pltmean,color=colours[i,:],alpha=1)
ax1.plot([0,0],[0,max(pltmean+pltse)],linestyle='--',color='k',zorder=3)
ax1.set_xlabel('Phase-heading offset (deg)')
ax1.set_xticks([-180,-90,0,90,180])
ax1.set_ylabel('Probability - EB')
ax1.set_title('Amenotaxis')
ax1.set_ylim([0,0.4])
ax2 = ax1.twinx()
ax2.set_ylabel('Probability - FSB')
for i in range(2):
    #plt.plot(pltbins,plt_et[:,i,:],color=colours[i,:],alpha=0.5)
    tdat = plt_meno[:,i+1,:]
    pltmean = np.mean(tdat,axis=1)
    pltse  = np.std(tdat,axis=1)/np.sqrt(len(datadirs))
    ax2.fill_between(pltbins,pltmean-pltse,pltmean+pltse,color=colours[i+1,:],alpha=0.5)
    ax2.plot(pltbins,pltmean,color=colours[i+1,:],alpha=1)
    #
ax2.set_ylim([0,0.25])
fig.tight_layout()  
plt.show()
plt.savefig(os.path.join(savedir,'AmenotaxisHisto2.pdf'))
#%% Point to point heatmaps
# Plume return
plt.close('all')
savedir = "Y:\\Presentations\\ForVanessa"
cxa.point2point_heat(2500,3000,toffset=0,arrowpoint=np.array([130,243,375]))
g = plt.gcf()
g.set_figheight(9.66)
plt.savefig(os.path.join(savedir,'ReturnLongTrace.pdf'))
# Plume exit
cxa.point2point_heat(2500,3000,toffset=0,arrowpoint=np.array([10,172,277]))
g = plt.gcf()
g.set_figheight(9.66)
plt.savefig(os.path.join(savedir,'LeaveLongTrace.pdf'))
# Amenotaxis
cxa.point2point_heat(0,500,toffset=0,arrowpoint=np.array([100,250,390]))
g = plt.gcf()
g.set_figheight(9.66)
plt.savefig(os.path.join(savedir,'AmenotaxisLongTrace.pdf'))
#%% Point return heatmaps for HHW
savedir = "Y:\\Applications\\HHW\\Figures\\Figure1"
cxa.point2point_heat(2500,3000,toffset=0,arrowpoint=np.array([130,243,375]))
plt.savefig(os.path.join(savedir,'ReturnLongTrace.pdf'))
#%%
# Determine jumps and jump direction
ft2 = cxa.ft2
pv2 = cxa.pv2
jumps = ft2['jump']
ins = ft2['instrip']
x = ft2['ft_posx'].to_numpy()
y = ft2['ft_posy'].to_numpy()
times = pv2['relative_time']
x,y = cxa.fictrac_repair(x,y)
insd = np.diff(ins)
ents = np.where(insd>0)[0]+1
exts = np.where(insd<0)[0]+1 
jd = np.diff(jumps)
jn = np.where(np.abs(jd)>0)[0]
jkeep = np.where(np.diff(jn)>1)[0]
jn = jn[jkeep]
jns = np.sign(jd[jn])
phase = cxa.pdat['offset_fsb_upper_phase']
#phase = ft2['ft_heading']
amp = cxa.pdat['amp_fsb_upper']
for i,j in enumerate(jn):
    # Plot all these trajs
    ex = exts-j
    ie = np.argmin(np.abs(ex))
    t_ent = ie+1
    sub_dx = exts[ie]
    tdx = np.arange(ents[ie],ents[t_ent],step=1,dtype='int')
    if jns[i]==-1:
        plt.figure(1)
        plt.plot(x[tdx]-x[sub_dx]+i*10,y[tdx]-y[sub_dx],color='k')
    elif jns[i]==1:
        plt.figure(2)
        plt.plot(x[tdx]-x[sub_dx]+i*10,y[tdx]-y[sub_dx],color='k')
    tdx2 = tdx[np.arange(0,len(tdx),10,dtype='int')]
    dt = times[tdx[-1]]-times[sub_dx]
    print(dt)
    for t in tdx2:
        xa = 50*amp[t]*np.sin(phase[t])
        ya = 50*amp[t]*np.cos(phase[t])
        plt.arrow(x[t]-x[sub_dx]+i*10,y[t]-y[sub_dx],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
plt.figure(1)    
plt.title('Jumps: Plume on the left')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

plt.figure(2)    
plt.title('Jumps: Plume on the right')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()
#%% Plot average traj and arrows
from scipy.stats import circmean, circstd
time_threshold = 60
# Pick the most common side
v,c = np.unique(jns,return_counts=True)
side = v[np.argmax(c)]
# Get time of return: choose quick returns
dt = []
for i,j in enumerate(jn):
    ex = exts-j
    ie = np.argmin(np.abs(ex))
    t_ent = ie+1
    sub_dx = exts[ie]
    tdx = np.arange(ents[ie],ents[t_ent],step=1,dtype='int')
    dt.append(times[tdx[-1]]-times[sub_dx])


this_j = jn[np.logical_and(jns==side, np.array(dt)<time_threshold)]

# Initialise arrays
inplume_traj = np.zeros((50,len(this_j),2))
outplume_traj = np.zeros((50,len(this_j),2))

outplume_phase = np.zeros((50,len(this_j),3))
inplume_phase = np.zeros((50,len(this_j),3))
outplume_amp = np.zeros((50,len(this_j),3))
inplume_amp = np.zeros((50,len(this_j),3))
side_mult = side*-1
x = ft2['ft_posx'].to_numpy()
y = ft2['ft_posy'].to_numpy()
x,y = cxa.fictrac_repair(x,y)
plt.fill([-10,0,0,-10],[-50,-50,0,0],color=[0.8,0.8,0.8])
plt.plot([0,0],[-50,0],linestyle='--',color='k',zorder=1)
plt.fill([-13,-3,-3,-13],[50,50,0,0],color=[0.8,0.8,0.8])
plt.plot([-3,-3],[50,0],linestyle='--',color='k',zorder=2)
x = x*side_mult
for i,j in enumerate(this_j):
    ex = exts-j
    ie = np.argmin(np.abs(ex))
    t_ent = ie+1
    sub_dx = exts[ie]
    phase = cxa.pdat['offset_eb_phase'].to_numpy()
    phase = phase.reshape(-1,1)
    phase = np.append(phase,cxa.pdat['offset_fsb_upper_phase'].to_numpy().reshape(-1,1),axis=1)
    phase = np.append(phase,cxa.pdat['offset_fsb_lower_phase'].to_numpy().reshape(-1,1),axis=1)
    phase = phase*side_mult
    amp = cxa.pdat['amp_eb']
    amp = amp.reshape(-1,1)
    amp = np.append(amp,cxa.pdat['amp_fsb_upper'].reshape(-1,1),axis=1)
    amp = np.append(amp,cxa.pdat['amp_fsb_lower'].reshape(-1,1),axis=1)
    # in plume
    ipdx = np.arange(ents[ie],sub_dx,step=1,dtype=int)
    old_time = ipdx-ipdx[0]
    ip_x = x[ipdx]
    ip_y = y[ipdx]
    ip_x = ip_x-ip_x[-1]
    ip_y = ip_y-ip_y[-1]
    new_time = np.linspace(0,max(old_time),50)
    x_int = np.interp(new_time,old_time,ip_x)
    y_int = np.interp(new_time,old_time,ip_y)
    inplume_traj[:,i,0] = x_int
    inplume_traj[:,i,1] = y_int
    for p in range(3):
        t_p = phase[ipdx,p]
        p_int = np.interp(new_time,old_time,t_p)
        inplume_phase[:,i,p] = p_int
        
        t_a = amp[ipdx,p]
        a_int = np.interp(new_time,old_time,t_a)
        inplume_amp[:,i,p] = a_int
    
    
    plt.plot(x_int,y_int,color='r',alpha=0.1,zorder=3)
    
    # out plume
    ipdx = np.arange(sub_dx,ents[t_ent],step=1,dtype=int)
    old_time = ipdx-ipdx[0]
    ip_x = x[ipdx]
    ip_y = y[ipdx]
    ip_x = ip_x-ip_x[0]
    ip_y = ip_y-ip_y[0]
    new_time = np.linspace(0,max(old_time),50)
    x_int = np.interp(new_time,old_time,ip_x)
    y_int = np.interp(new_time,old_time,ip_y)
    outplume_traj[:,i,0] = x_int
    outplume_traj[:,i,1] = y_int
    for p in range(3):
        t_p = phase[ipdx,p]
        p_int = np.interp(new_time,old_time,t_p)
        outplume_phase[:,i,p] = p_int
        
        t_a = amp[ipdx,p]
        a_int = np.interp(new_time,old_time,t_a)
        outplume_amp[:,i,p] = a_int
    
    plt.plot(x_int,y_int,color='k',alpha=0.1)
    
    tdx = np.arange(ents[ie],ents[t_ent],step=1,dtype='int')
    
inmean_traj = np.mean(inplume_traj,axis=1)
outmean_traj = np.mean(outplume_traj,axis=1)
inmean_phase = circmean(inplume_phase,high=np.pi,low=-np.pi,axis=1)
outmean_phase = circmean(outplume_phase,high=np.pi,low=-np.pi,axis=1)
inmean_amp = np.mean(inplume_amp,axis=1)
outmean_amp = np.mean(outplume_amp,axis=1)

plt.plot(inmean_traj[:,0],inmean_traj[:,1],color='r',zorder=4)
plt.plot(outmean_traj[:,0],outmean_traj[:,1],color='k',zorder=4)
colours = np.array([[0.3,0.3,0.3],[0.3,0.3,1],[0.8,0.3,1]])

tdx2 = np.arange(0,50,step=10,dtype=int)
for p in range(3):
    for t in tdx2:
        xa = 50*inmean_amp[t,p]*np.sin(inmean_phase[t,p])
        ya = 50*inmean_amp[t,p]*np.cos(inmean_phase[t,p])
        plt.arrow(inmean_traj[t,0],inmean_traj[t,1],xa,ya,length_includes_head=True,head_width=1,color=colours[p,:],zorder=5)
        
        xa = 50*outmean_amp[t,p]*np.sin(outmean_phase[t,p])
        ya = 50*outmean_amp[t,p]*np.cos(outmean_phase[t,p])
        plt.arrow(outmean_traj[t,0],outmean_traj[t,1],xa,ya,length_includes_head=True,head_width=1,color=colours[p,:],zorder=5)
    xa = 50*outmean_amp[-1,p]*np.sin(outmean_phase[-1,p])
    ya = 50*outmean_amp[-1,p]*np.cos(outmean_phase[-1,p])
    plt.arrow(outmean_traj[-1,0],outmean_traj[-1,1],xa,ya,length_includes_head=True,head_width=1,color=colours[p,:],zorder=5)
yl = [np.min(inmean_traj[:,1])-10,np.max(outmean_traj[:,1])+10]
plt.ylim(yl)
plt.xlim([-10,10])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()
# Get entry after jump
# Check dTime between prior exit and jump

# Look at the mean



# %%
# 1 time lagged correlation of phase between FC2 and heading

# 2 Plot trajectories where FC2 points towards the plume edge on return

pv2 = cxa.pv2
ft2 = cxa.ft2
ins = ft2['instrip'].to_numpy()
insd = np.diff(ins)
entries = np.where(insd>0)[0]+1 
exits = np.where(insd<0)[0]+1
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()

xs = np.where(ins==1)[0][0]
x = x-x[xs]
y = y-y[xs]
x,y = cxa.fictrac_repair(x,y)
plume_cent = 0
to_plume_upper = []
to_plume_lower = []
for i,e in enumerate(entries):
    p1 = cxa.pdat['offset_fsb_upper_phase'][e]
    p2 = cxa.pdat['offset_fsb_lower_phase'][e]
    jump = ft2['jump'][e]
    pc = plume_cent+jump
    tx = x[e]
    r = np.sign(tx-pc)# right of centre is positive
    ps = p1*r
    if np.sign(ps)<0:
        print('Points to plume with angle: ', (p1/np.pi)*180)
        to_plume_upper.append(int(e))
    else:
        print('Points away from plume with angle', (p1/np.pi)*180)
            
    ps = p2*r
    if np.sign(ps)<0:
        print('Points to plume with angle: ', (p1/np.pi)*180)
        to_plume_lower.append(e) 
    else:
        print('Points away from plume with angle', (p1/np.pi)*180)


# plot from prior entry to this entry
for i,e in enumerate(to_plume_upper):
    ef = np.where(entries==e)[0]
    ef_st = np.max([ef[0]-1,0])
    ef_ed = np.min([ef[0]+1,len(entries)])
    plt.figure()
    st = exits[ef_st]
    ed = entries[ef_ed]
    dx = np.arange(st,ed,1)
    plt.plot(x[dx],y[dx])
    xs = x[dx]
    ys = y[dx]
    iss = ins[dx]
    plt.scatter(xs[iss>0],ys[iss>0],color=[0.5,0.5,0.5])
    
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()