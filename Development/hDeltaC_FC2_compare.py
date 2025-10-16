# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 17:52:02 2025

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
plt.rcParams['font.sans-serif'] = 'Arial'
from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper
savedir = 'Y:\\Data\\FCI\\FCI_summaries\\FC2_hDeltaC_comparison'
#%% Load up data
datadirs_hdc = [
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250411\f1\Trial1",# Phase recording is not the best - strong pointer
                #r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial3",# Not many jumps, weak pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f2\Trial2", # Strong pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial2",# Strong pointer, just like FC2
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial1", # Points away ******
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial3", # Strong pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial2",# Strong pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250907\f1\Trial2",# Good pointer
                ]

all_flies_hdc = {}
etp_hdc = {}
for i,datadir in enumerate(datadirs_hdc):
    print(datadir)
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    all_flies_hdc.update({str(i):cxa})
    etp = ET_paper(datadir)
    etp_hdc.update({str(i):etp})

datadirs_fc2 = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial5",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial2",
r"Y:\Data\FCI\Hedwig\FC2_maimon2\250128\f1\Trial4"]
all_flies_fc2 = {}
etp_fc2 = {}
for i,datadir in enumerate(datadirs_fc2):
    print(datadir)
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    all_flies_fc2.update({str(i):cxa})
    etp = ET_paper(datadir)
    etp_fc2.update({str(i):etp})
    
datadirs_fc2_pam = [r"Y:\Data\FCI\Hedwig\FC2_PAM\250805\f2\Trial2",
                r"Y:\Data\FCI\Hedwig\FC2_PAM\250806\f1\Trial2"]
all_flies_fc2pam = {}
etp_fc2pam = {}
for i,datadir in enumerate(datadirs_fc2_pam):
    print(datadir)
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False,stim=True)
    all_flies_fc2pam.update({str(i):cxa})
    etp = ET_paper(datadir)
    etp_fc2pam.update({str(i):etp})
    
datadirs_hdj = [
                r"Y:\Data\FCI\Hedwig\hDeltaJ\240529\f1\Trial3",#Good pointer
                ]

all_flies_hdj = {}
etp_hdj = {}
for i,datadir in enumerate(datadirs_hdj):
    print(datadir)
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    all_flies_hdj.update({str(i):cxa})
    etp = ET_paper(datadir)
    etp_hdj.update({str(i):etp})
    
#%% Pre vs post reinforcement/threshold
savedir = r'Y:\Data\FCI\FCI_summaries\FC2_PAM'
datadirs = [r"Y:\Data\FCI\Hedwig\FC2_PAM\250805\f2\Trial2",
 r"Y:\Data\FCI\Hedwig\FC2_PAM\250806\f1\Trial2"]
fignames = ['PVA','mean_fluor']
diff = True
for d in all_flies_fc2:
    plt.close('all')
    cxa = all_flies_fc2[d]
    #cxa = CX_a(d,regions=['eb','fsb_upper','fsb_lower'],yoking=True,stim=True,denovo=False)
    e_e = cxa.get_entries_exits_like_jumps()
    jumps = cxa.get_jumps(time_threshold=1000000)
    led = cxa.ft2['led1_stpt'].to_numpy()
    ledon = np.where(led==0)[0][0]-10
    if diff:
        phase = ug.circ_subtract(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),cxa.pdat['offset_eb_phase'].to_numpy())
    else:
        phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    amp = ug.get_pvas(cxa.pdat['wedges_fsb_upper'])
    amp2 = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
    eon = np.where(e_e[:,2]>ledon)[0][0]-0.5
    
    plt.figure(1,figsize=(9,3))
    plt.plot([0,len(e_e)],[0,0],color='k',linestyle='--')
    plt.plot([0,len(e_e)],[-np.pi/2,-np.pi/2],color='k',linestyle='--')
    plt.plot([0,len(e_e)],[np.pi/2,np.pi/2],color='k',linestyle='--')
    
    plt.plot([eon,eon],[-np.pi,np.pi],color='r',linestyle='-')
    plt.figure(2,figsize=(9,3))
    plt.plot([0,len(e_e)],[0,0],color='k',linestyle='--')
    plt.plot([0,len(e_e)],[-np.pi/2,-np.pi/2],color='k',linestyle='--')
    plt.plot([0,len(e_e)],[np.pi/2,np.pi/2],color='k',linestyle='--')
    
    plt.plot([eon,eon],[-np.pi,np.pi],color='r',linestyle='-')
    
    amp = amp-np.min(amp)
    amp[amp>np.percentile(amp,95)] = np.percentile(amp,95)
    amp = amp/np.max(amp)
    amp = np.round(amp*99).astype(int)
    
    amp2 = amp2-np.min(amp2)
    amp2[amp2>np.percentile(amp2,95)] = np.percentile(amp2,95)
    amp2 = amp2/np.max(amp2)
    amp2 = np.round(amp2*99).astype(int)
    
    cmap = plt.get_cmap('viridis')
    # Sample 100 evenly spaced points from the colormap
    colours = cmap(np.linspace(0, 1, 100))
    
    # Drop the alpha channel -> get 100x3 array
    rgb_array = colours[:, :3]
    for i,e in enumerate(e_e): 
        dx = np.arange(e[1],e[-1])
        if len(dx)>5:
            dx = dx[-5:]
            
        tp = stats.circmean(phase[dx],high=np.pi,low=-np.pi)
        ta = np.mean(amp[dx]).astype(int)
        ta2 = np.mean(amp2[dx]).astype(int)
        
        plt.figure(1)
        if np.sum(jumps[:,2]==e[2])>0:
            plt.scatter(i,tp,color=rgb_array[ta,:],marker='*')
        else:
            plt.scatter(i,tp,color=rgb_array[ta,:])
            
        plt.figure(2)
        if np.sum(jumps[:,2]==e[2])>0:
            plt.scatter(i,tp,color=rgb_array[ta2,:],marker='*')
        else:
            plt.scatter(i,tp,color=rgb_array[ta2,:])
        
        
    for i in range(2):
        plt.figure(i+1)
        plt.xlabel('entry number')
        plt.subplots_adjust(bottom=0.3)
        if diff:
            plt.ylabel('FC2 - EB phase')
            plt.savefig(os.path.join(savedir,fignames[i] +'_diff_'+ cxa.name+'.png'))
        else:
            plt.ylabel('FC2 phase')
   
            plt.savefig(os.path.join(savedir,fignames[i] + cxa.name+'.png'))
#%% regression modelling of wedges
import sklearn.linear_model as lm
plt.close('all')
ymat = cxa.pdat['wedges_fsb_upper']
wed_eb = cxa.pdat['wedges_eb']
eb = cxa.pdat['phase_eb'][:,np.newaxis]
ebmat = np.tile(eb,(1,16))
fsb = cxa.pdat['phase_fsb_upper'][:,np.newaxis]
fsbmat = np.tile(fsb,(1,16))
tscale = 10
wedid = np.linspace(-np.pi,np.pi,16)
tp = eb[100]
eb_cos = np.cos(wedid+fsbmat)
eb_cos = eb_cos+1 
fsb_cos =np.cos(wedid+fsbmat-np.pi)
fsb_cos = fsb_cos+1
ymat = ymat-np.min(ymat,axis=1)[:,np.newaxis]
ymat = ymat/np.max(ymat,axis=1)[:,np.newaxis]
eb_delays = np.array([0,0.5,1,2,4,8])*tscale
#eb_delays = np.array([0])*tscale
fsb_delay = np.array([0.5,1,2,4,8])*tscale
eb_delays = eb_delays.astype(int)
fsb_delay = fsb_delay.astype(int)
start = np.max(eb_delays)
regdx = np.arange(start,len(ymat),1)

betas = np.zeros((len(regdx),len(eb_delays)+len(fsb_delay)))
r2s = np.zeros((len(regdx),len(eb_delays)+len(fsb_delay)))

# Change to do regression separately on each regressor since you get a good fit every time
for i,r in enumerate(regdx):
    print(i)
    xdx = i-eb_delays
    xeb = eb_cos[xdx,:]
    xdxf = i-fsb_delay
    xfsb = fsb_cos[xdxf,:]
    X = np.append(xeb,xfsb,axis=0)
    X = np.transpose(X)
    
    y = ymat[i,:]
    for a in range(X.shape[1]):
        reg = lm.LinearRegression(fit_intercept=False)
        x = X[:,a][:,np.newaxis]
        reg.fit(x,y);
        betas[i,a] = reg.coef_
        r2s[i,a] = reg.score(x,y)
    # 
    # plt.figure()
    # plt.plot(y)
    # plt.plot(reg.predict(X))
    
    
plt.imshow(r2s,aspect='auto',interpolation='none',vmax=0.5,vmin=0)

#%%
x = regdx
plt.scatter(x,fsb[x],color='r',s=5)
plt.plot((r2s[:,6]>0.25) *3)
plt.scatter(x,ug.circ_subtract(fsb[x],-np.pi),color=[1,0.5,0.5],s=5)
#%% Inferrred goal vs phase
plt.close('all')
grand_mean_fc2 = np.zeros((len(all_flies_fc2),2))
gm_diff_fc2 = np.zeros(len(all_flies_fc2))
mode ='jumps'
for i,f in enumerate(all_flies_fc2):
    cxa= all_flies_fc2[f]
    if mode=='all':
        e_e = cxa.get_entries_exits_like_jumps()
    elif mode=='jumps':
        e_e = cxa.get_jumps()
    fsb_phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    heading = cxa.ft2['ft_heading']
    plt.figure()
    #plt.plot([0,len(e_e)],[0,0],color='k')
    plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='r')
    plt.plot([-np.pi,0],[0,np.pi],color='r')
    plt.plot([0,np.pi],[-np.pi,0],color='r')
    data_goal = np.array([])
    data_inf_goal = np.array([])
    data_diff = np.array([])
    data_ent_ang = np.array([])
    for ie,e in enumerate(e_e[1:]):
        if mode=='all':
            if (e[0]-e_e[ie,1])<30:
                continue
            if (e[2]-e[1])<30:
                continue
        strt =np.max([e[0]-5,e_e[ie,1]])
        dx = np.arange(strt,e[0])
        inf_goal = stats.circmean(heading[dx],low=-np.pi,high=np.pi)
        data_inf_goal = np.append(data_inf_goal,inf_goal)
        strt = np.max([e[2]-5,e[1]])
        dx = np.arange(strt,e[2])
        goal = stats.circmean(fsb_phase[dx],low=-np.pi,high=np.pi)
        entang = stats.circmean(heading[dx],low=-np.pi,high=np.pi)
        data_goal = np.append(data_goal,goal)
        data_diff = np.append(data_diff,goal-inf_goal)
        data_ent_ang = np.append(data_ent_ang,entang)
        #plt.scatter(ie,goal,color='b')
        #plt.scatter(ie,inf_goal,color='r')
    igm = stats.circmean(data_inf_goal,high=np.pi,low=-np.pi)
    
    
    x = np.arange(0,len(data_goal))
    #plt.scatter(x,data_inf_goal,color='r',s=5)
    #plt.scatter(x,data_goal,color='b',s=5)
    #plt.plot([0,x[-1]],[igm,igm],color='r')
    #plt.plot([0,x[-1]],[0,0],color='k')
    plt.scatter(data_inf_goal,data_goal,color='r')
    plt.scatter(data_ent_ang,data_goal,color='k')
    gm = stats.circmean(data_goal,high=np.pi,low=-np.pi)
    #plt.plot([0,x[-1]],[gm,gm],color='b')
    plt.ylim([-np.pi,np.pi])
    plt.xlim([-np.pi,np.pi])
    grand_mean_fc2[i,:] = [igm,gm]
    gm_diff_fc2[i] = stats.circmean(data_diff,low=-np.pi,high=np.pi)
    
grand_mean_hdc = np.zeros((len(all_flies_hdc),2))
gm_diff_hdc = np.zeros(len(all_flies_fc2))
for i, f in enumerate(all_flies_hdc):
    cxa= all_flies_hdc[f]
    if mode=='all':
        e_e = cxa.get_entries_exits_like_jumps()
    elif mode=='jumps':
        e_e = cxa.get_jumps()
    fsb_phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    heading = cxa.ft2['ft_heading']
    plt.figure()
    plt.plot([0,len(e_e)],[0,0],color='k')
    data_goal = np.array([])
    data_inf_goal = np.array([])
    data_diff = np.array([])
    for ie,e in enumerate(e_e[1:]):
        if mode =='all':
            if (e[0]-e_e[ie,1])<30:
                continue
            if (e[2]-e[1])<30:
                continue
        
        strt =np.max([e[0]-5,e_e[ie,1]])
        dx = np.arange(strt,e[0])
        inf_goal = stats.circmean(heading[dx],low=-np.pi,high=np.pi)
        data_inf_goal = np.append(data_inf_goal,inf_goal)
        strt = np.max([e[2]-5,e[1]])
        dx = np.arange(strt,e[2])
        goal = stats.circmean(fsb_phase[dx],low=-np.pi,high=np.pi)
        data_goal = np.append(data_goal,goal)
        plt.scatter(ie,goal,color='b')
        plt.scatter(ie,inf_goal,color='r')
        data_diff = np.append(data_diff,goal-inf_goal)
    igm = stats.circmean(data_inf_goal,high=np.pi,low=-np.pi)
    plt.plot([0,ie],[igm,igm],color='r')
    
    gm = stats.circmean(data_goal,high=np.pi,low=-np.pi)
    plt.plot([0,ie],[gm,gm],color='b')
    plt.ylim([-np.pi,np.pi])
    grand_mean_hdc[i,:] = [igm,gm]
    gm_diff_hdc[i] = stats.circmean(data_diff,low=-np.pi,high=np.pi)
    
    
    
    


plt.figure()
grand_mean_fc2 = 180*grand_mean_fc2/np.pi
grand_mean_hdc = 180*grand_mean_hdc/np.pi
gm_diff_fc2 = gm_diff_fc2*-1*np.sign(grand_mean_fc2[:,0])
grand_mean_fc2 = -1*grand_mean_fc2*np.sign(grand_mean_fc2[:,0])[:,np.newaxis]
x = [0,1]
plt.plot(x,np.transpose(grand_mean_fc2),color='k',alpha=0.5);
gmn = stats.circmean(grand_mean_fc2,high=180,low=-180,axis=0)
plt.plot(x,gmn,color='k',linewidth=3)

grand_mean_hdc = -1*grand_mean_hdc*np.sign(grand_mean_hdc[:,0])[:,np.newaxis]
x = [1.5,2.5]
plt.plot(x,np.transpose(grand_mean_hdc),color='k',alpha=0.5);
plt.ylim([-180,180])
plt.yticks([-180,-90,0,90,180])
gmn = stats.circmean(grand_mean_hdc,high=180,low=-180,axis=0)
plt.plot(x,gmn,color='k',linewidth=3)
plt.xticks([0,1,1.5,2.5],labels=['FC2 inferred goal','FC2 phase','HdC inferred goal','HdC phase'],rotation=45)
plt.subplots_adjust(bottom=.2)

plt.figure()
r = 1
idx = np.argsort(-gm_diff_fc2)
for i in idx:
    tig = grand_mean_fc2[i,0]
    tg = grand_mean_fc2[i,1]
    thetas = np.linspace(tig,tg,100)
    xs =r* np.sin(np.pi*thetas/180)
    ys = r*np.cos(np.pi*thetas/180)
    plt.plot(xs,ys,color='k',zorder=5)
    r = r+0.05
    plt.scatter(xs[0],ys[0],color='r',zorder=10,s=5)
    plt.plot([0,xs[0]],[0,ys[0]],color='r',alpha=0.5)
    plt.scatter(xs[-1],ys[-1],color='b',zorder=10,s=5)
    plt.plot([0,xs[-1]],[0,ys[-1]],color='b',alpha=0.5)
    
#plt.xlim([-1.5,1.5])
#plt.ylim([-1.5,1.5])

r = 1
idx = np.argsort(-gm_diff_hdc)
for i in idx:
    tig = grand_mean_hdc[i,0]
    tg = grand_mean_hdc[i,1]
    thetas = np.linspace(tig,tg,100)
    xs =r* np.sin(np.pi*thetas/180)+2.5
    ys = r*np.cos(np.pi*thetas/180)
    plt.plot(xs,ys,color='k',zorder=5)
    r = r+0.05
    plt.scatter(xs[0],ys[0],color='r',zorder=10,s=5)
    plt.plot([2.5,xs[0]],[0,ys[0]],color='r',alpha=0.5)
    plt.scatter(xs[-1],ys[-1],color='b',zorder=10,s=5)
    plt.plot([2.5,xs[-1]],[0,ys[-1]],color='b',alpha=0.5)
    
#plt.xlim([-2,3])
#plt.ylim([-2,3])
g = plt.gca()
g.set_aspect('equal')
plt.text(0,1.3,'FC2',horizontalalignment='center')
plt.text(2.5,1.3,'hDeltaC',horizontalalignment='center')
plt.xticks([])
plt.yticks([])
plt.savefig(os.path.join(savedir,'Inferred_goal_goal_comp.png'))
#%%
plt.figure()
x =np.sin(np.pi*grand_mean_fc2/180)
y = np.cos(np.pi*grand_mean_fc2/180)
xplt = np.zeros(x.shape)
xplt[:,1] =x[:,0]
yplt = np.zeros(y.shape)
yplt[:,1] = y[:,0]
plt.plot(np.transpose(xplt),np.transpose(yplt),color='r')

xplt = np.zeros(x.shape)
xplt[:,1] =x[:,1]
yplt = np.zeros(y.shape)
yplt[:,1] = y[:,1]
plt.plot(np.transpose(xplt),np.transpose(yplt),color='b')

xplt = np.zeros(x.shape)
xplt[:,1] = np.sin(gm_diff_fc2)
yplt = np.zeros(y.shape)
yplt[:,1] = np.cos(gm_diff_fc2)
plt.plot(np.transpose(xplt),np.transpose(yplt),color='k')



x =np.sin(np.pi*grand_mean_hdc/180)+1.5
y = np.cos(np.pi*grand_mean_hdc/180)
xplt = np.zeros(x.shape)+1.5
xplt[:,1] =x[:,0]
yplt = np.zeros(y.shape)
yplt[:,1] = y[:,0]
plt.plot(np.transpose(xplt),np.transpose(yplt),color='r')

xplt = np.zeros(x.shape)+1.5
xplt[:,1] =x[:,1]
yplt = np.zeros(y.shape)
yplt[:,1] = y[:,1]
plt.plot(np.transpose(xplt),np.transpose(yplt),color='b')
g = plt.gca()
g.set_aspect('equal')
plt.text(0,1.1,'FC2',horizontalalignment='center')
plt.text(1.5,1.1,'hDeltaC',horizontalalignment='center')
plt.ylim([-1.2,1.2])



#%% Phase transition plot
plt.close('all')
for i,datadir in enumerate(datadirs_fc2):
    plt.figure()
    cxa = all_flies_fc2[str(i)]
    etp = etp_fc2[str(i)]
    phase_eb = cxa.pdat['offset_eb_phase'].to_numpy()
    phases,traj = etp.trajectory_mean(regions=['eb','fsb_upper'],bins=10)
    # phases = ug.circ_subtract(phases,phases[10,0,:])
    phase_fsb = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    jumps = cxa.get_jumps()
    pmean = stats.circmean(phases,low=-np.pi,high=np.pi,axis=2)
    plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
    plt.plot([-np.pi,0],[0,np.pi],color='k',linestyle='--')
    plt.plot([0,np.pi],[-np.pi,0],color='k',linestyle='--')
    
    # for j in jumps:
    #     plt.plot(phase_eb[j[0]:j[2]],phase_fsb[j[0]:j[2]],color='k',alpha=0.25)
    plt.plot(pmean[:10,0],pmean[:10,1],color='r')
    plt.plot(pmean[9:,0],pmean[9:,1],color=[0.3,0.3,1])
    plt.xlim([-np.pi,np.pi])
    plt.ylim([-np.pi,np.pi])
    plt.xlabel('EPG phase')
    plt.ylabel('FC2 phase')
    
for i,datadir in enumerate(datadirs_hdc):
    plt.figure()
    cxa = all_flies_hdc[str(i)]
    etp = etp_hdc[str(i)]
    phase_eb = cxa.pdat['offset_eb_phase'].to_numpy()
    phases,traj = etp.trajectory_mean(regions=['eb','fsb_upper'],bins=10)
    phase_fsb = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    jumps = cxa.get_jumps()
    pmean = stats.circmean(phases,low=-np.pi,high=np.pi,axis=2)
    plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
    plt.plot([-np.pi,0],[0,np.pi],color='k',linestyle='--')
    plt.plot([0,np.pi],[-np.pi,0],color='k',linestyle='--')
    
    # for j in jumps:
    #     plt.plot(phase_eb[j[0]:j[2]],phase_fsb[j[0]:j[2]],color='k',alpha=0.25)
    plt.plot(pmean[:10,0],pmean[:10,1],color='r')
    plt.plot(pmean[9:,0],pmean[9:,1],color=[0.3,0.3,1])
    plt.xlim([-np.pi,np.pi])
    plt.ylim([-np.pi,np.pi])
    plt.xlabel('EPG phase')
    plt.ylabel('hDeltaC phase')

#%% Stop start phase scatter
savedir = r'Y:\Data\FCI\FCI_summaries\hDeltaC'
for i,datadir in enumerate(all_flies_hdc):
    cxa = all_flies_hdc[str(i)]
    cxa.cxa_stop_start_phase_scatter()
    for i in range(3):
        plt.figure(i)
        plt.savefig(os.path.join(savedir,str(i) + 'Mv_Stp_' +cxa.name+'.png'))
        plt.figure(i+100)
        plt.savefig(os.path.join(savedir,str(i) + 'Stp_Mv_' +cxa.name+'.png'))
savedir = r'Y:\Data\FCI\FCI_summaries\FC2_maimon2'
for i,datadir in enumerate(datadirs_fc2):
    cxa = all_flies_fc2[str(i)]
    cxa.cxa_stop_start_phase_scatter()
    for i in range(3):
        plt.figure(i)
        plt.savefig(os.path.join(savedir,str(i) + 'Mv_Stp_' +cxa.name+'.png'))
        plt.figure(i+100)
        plt.savefig(os.path.join(savedir,str(i) + 'Stp_Mv_' +cxa.name+'.png'))
savedir = r'Y:\Data\FCI\FCI_summaries\hDeltaJ'
for i,datadir in enumerate(all_flies_hdj):
    cxa = all_flies_hdj[str(i)]
    cxa.cxa_stop_start_phase_scatter()
    for i in range(3):
        plt.figure(i)
        plt.savefig(os.path.join(savedir,str(i) + 'Mv_Stp_' +cxa.name+'.png'))
        plt.figure(i+100)
        plt.savefig(os.path.join(savedir,str(i) + 'Stp_Mv_' +cxa.name+'.png'))
#%%
plt.figure()
plt.subplot(2,2,1)
ss_mean = np.zeros((len(all_flies_fc2),2,2))
minsize = 20
condition = 'last'
for i,datadir in enumerate(datadirs_fc2):
    cxa = all_flies_fc2[str(i)]
    stop_starts = cxa.stop_start_jumps(minsize=minsize,condition=condition)
    ss_mean[i,:,:] = stats.circmean(stop_starts,high=np.pi,low=-np.pi,axis=0,nan_policy='omit')
    
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
plt.plot([-np.pi,0],[0,np.pi],color='k',linestyle='--')
plt.plot([0,np.pi],[-np.pi,0],color='k',linestyle='--')
plt.plot([-np.pi/2,-np.pi/2],[-np.pi,np.pi],color='r',linestyle='--')
plt.plot([-np.pi,np.pi],[-np.pi/2,-np.pi/2],color='r',linestyle='--')
plt.scatter(ss_mean[:,0,0,],ss_mean[:,0,1],color=[0,0,0.6],s=10,alpha=0.5)
plt.scatter(ss_mean[:,1,0,],ss_mean[:,1,1],color=[0,0.6,0.9],s=10,alpha=0.5)
all_mn = stats.circmean(ss_mean,axis=0,high=np.pi,low=-np.pi)
plt.scatter(all_mn[0,0,],all_mn[0,1],color=[0,0,0.6],s=100)
plt.scatter(all_mn[1,0,],all_mn[1,1],color=[0,0.6,0.9],s=100)
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.title('FC2')
plt.subplot(2,2,2)
ss_mean = np.zeros((len(all_flies_hdc),2,2))
for i,datadir in enumerate(datadirs_hdc):
    
    cxa = all_flies_hdc[str(i)]
    stop_starts = cxa.stop_start_jumps(minsize=minsize,condition=condition)
    ss_mean[i,:,:] = stats.circmean(stop_starts,high=np.pi,low=-np.pi,axis=0,nan_policy='omit')
    
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
plt.plot([-np.pi,0],[0,np.pi],color='k',linestyle='--')
plt.plot([0,np.pi],[-np.pi,0],color='k',linestyle='--')
plt.plot([-np.pi/2,-np.pi/2],[-np.pi,np.pi],color='r',linestyle='--')
plt.plot([-np.pi,np.pi],[-np.pi/2,-np.pi/2],color='r',linestyle='--')
plt.scatter(ss_mean[:,0,0,],ss_mean[:,0,1],color=[0,0,0.6],s=10,alpha=0.5)
plt.scatter(ss_mean[:,1,0,],ss_mean[:,1,1],color=[0,0.6,0.9],s=10,alpha=0.5)
all_mn = stats.circmean(ss_mean,axis=0,high=np.pi,low=-np.pi)

plt.scatter(all_mn[0,0,],all_mn[0,1],color=[0,0,0.6],s=100)
plt.scatter(all_mn[1,0,],all_mn[1,1],color=[0,0.6,0.9],s=100)
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.title('hDeltaC')
plt.subplot(2,2,3)
ss_mean = np.zeros((len(all_flies_hdj),2,2))
for i,datadir in enumerate(datadirs_hdj):
    
    cxa = all_flies_hdj[str(i)]
    stop_starts = cxa.stop_start_jumps(minsize=minsize,condition=condition)
    ss_mean[i,:,:] = stats.circmean(stop_starts,high=np.pi,low=-np.pi,axis=0,nan_policy='omit')
    
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
plt.plot([-np.pi,0],[0,np.pi],color='k',linestyle='--')
plt.plot([0,np.pi],[-np.pi,0],color='k',linestyle='--')
plt.plot([-np.pi/2,-np.pi/2],[-np.pi,np.pi],color='r',linestyle='--')
plt.plot([-np.pi,np.pi],[-np.pi/2,-np.pi/2],color='r',linestyle='--')
plt.scatter(ss_mean[:,0,0,],ss_mean[:,0,1],color=[0,0,0.6],s=10,alpha=0.5)
plt.scatter(ss_mean[:,1,0,],ss_mean[:,1,1],color=[0,0.6,0.9],s=10,alpha=0.5)
all_mn = stats.circmean(ss_mean,axis=0,high=np.pi,low=-np.pi)

plt.scatter(all_mn[0,0,],all_mn[0,1],color=[0,0,0.6],s=100)
plt.scatter(all_mn[1,0,],all_mn[1,1],color=[0,0.6,0.9],s=100)
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.title('hDeltaJ')
plt.ylabel('Phase fsb')
plt.xlabel('Phase eb')
#%%
plt.close('all')
for i,datadir in enumerate(datadirs_fc2):
    cxa = all_flies_fc2[str(i)]
    cxa.stop_start_transition(minsize=20)
#%% Stop start phase pinwheel
plt.figure()
plt.subplot(2,2,1)
ss_mean = np.zeros((len(all_flies_fc2),2,2))
minsize = 20
condition = 'last'
for i,datadir in enumerate(datadirs_fc2):
    cxa = all_flies_fc2[str(i)]
    stop_starts = cxa.stop_start_jumps(minsize=minsize,condition=condition)
    ss_mean[i,:,:] = stats.circmean(stop_starts,high=np.pi,low=-np.pi,axis=0,nan_policy='omit')

plt.plot([-1,1],[0,0],color='k',linestyle='--')
plt.plot([0,0],[-1,1],color='k',linestyle='--')
x = np.zeros((2,len(all_flies_fc2)))    
x[0,:] = np.sin(ss_mean[:,0,1])
y = np.zeros((2,len(all_flies_fc2)))    
y[0,:] = np.cos(ss_mean[:,0,1])

plt.plot(x,y,color=[0,0,0.6],alpha=0.3)
plt.plot(np.mean(x,axis=1),np.mean(y,axis=1),color=[0,0,0.6],linewidth=3)
x = np.zeros((2,len(all_flies_fc2)))    
x[0,:] = np.sin(ss_mean[:,1,1])
y = np.zeros((2,len(all_flies_fc2)))    
y[0,:] = np.cos(ss_mean[:,1,1])

plt.plot(x,y,color=[0,0.6,0.9],alpha=0.3)
plt.plot(np.mean(x,axis=1),np.mean(y,axis=1),color=[0,0.5,0.8],linewidth=3)
ss_mean = np.zeros((len(all_flies_hdc),2,2))

plt.title('FC2')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,2)
for i,datadir in enumerate(datadirs_hdc):
    
    cxa = all_flies_hdc[str(i)]
    stop_starts = cxa.stop_start_jumps(minsize=minsize,condition=condition)
    ss_mean[i,:,:] = stats.circmean(stop_starts,high=np.pi,low=-np.pi,axis=0,nan_policy='omit')
plt.plot([-1,1],[0,0],color='k',linestyle='--')
plt.plot([0,0],[-1,1],color='k',linestyle='--')
x = np.zeros((2,len(all_flies_hdc)))    
x[0,:] = np.sin(ss_mean[:,0,1])
y = np.zeros((2,len(all_flies_fc2)))    
y[0,:] = np.cos(ss_mean[:,0,1])

plt.plot(x,y,color=[0,0,0.6],alpha=0.3)
plt.plot(np.mean(x,axis=1),np.mean(y,axis=1),color=[0,0,0.6],linewidth=3)
x = np.zeros((2,len(all_flies_hdc)))    
x[0,:] = np.sin(ss_mean[:,1,1])
y = np.zeros((2,len(all_flies_hdc)))    
y[0,:] = np.cos(ss_mean[:,1,1])

plt.plot(x,y,color=[0,0.5,0.8],alpha=0.3)
plt.plot(np.mean(x,axis=1),np.mean(y,axis=1),color=[0,0.5,0.8],linewidth=3)

plt.title('hDeltaC')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,3)
ss_mean = np.zeros((len(all_flies_hdj),2,2))
for i,datadir in enumerate(datadirs_hdj):
    
    cxa = all_flies_hdj[str(i)]
    stop_starts = cxa.stop_start_jumps(minsize=minsize,condition=condition)
    ss_mean[i,:,:] = stats.circmean(stop_starts,high=np.pi,low=-np.pi,axis=0,nan_policy='omit')
plt.plot([-1,1],[0,0],color='k',linestyle='--')
plt.plot([0,0],[-1,1],color='k',linestyle='--')    
x = np.zeros((2,len(all_flies_hdj)))    
x[0,:] = np.sin(ss_mean[:,0,1])
y = np.zeros((2,len(all_flies_fc2)))    
y[0,:] = np.cos(ss_mean[:,0,1])

plt.plot(x,y,color=[0,0,0.6],alpha=0.3)
plt.plot(np.mean(x,axis=1),np.mean(y,axis=1),color=[0,0,0.6],linewidth=3)
x = np.zeros((2,len(all_flies_hdj)))    
x[0,:] = np.sin(ss_mean[:,1,1])
y = np.zeros((2,len(all_flies_fc2)))    
y[0,:] = np.cos(ss_mean[:,1,1])

plt.plot(x,y,color=[0,0.5,0.8],alpha=0.3)
plt.plot(np.mean(x,axis=1),np.mean(y,axis=1),color=[0,0.5,0.8],linewidth=3)

plt.title('hDeltaJ')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xticks([])
plt.yticks([])
#%% Phase excersions
plt.close('all')
i =3
cxa = all_flies_hdc[str(i)]
etp = etp_hdc[str(i)]
phase_eb = cxa.pdat['phase_eb']
phases,traj = etp.trajectory_mean(regions=['eb','fsb_upper'],bins=10)
phase_fsb = cxa.pdat['phase_fsb_upper']
phase_fsb = ug.savgol_circ(phase_fsb,20,3)
phase_eb = ug.savgol_circ(phase_eb,20,3)
dfsb = ug.circ_subtract(phase_fsb[1:],phase_fsb[:-1])
deb = ug.circ_subtract(phase_eb[1:],phase_eb[:-1])
std_fsb = np.percentile(np.abs(dfsb),5)
std_eb = np.percentile(np.abs(deb),5)
deb2 =ug.boxcar_sum(deb,5)
#deb2[np.abs(deb2)<std_eb] = np.nan
dfsb2 =ug.boxcar_sum(dfsb,5)
#dfsb2[np.abs(dfsb2)<std_fsb] = np.nan
plt.plot(dfsb)
plt.plot(deb)
jumps = cxa.get_jumps()
for j in jumps:
    plt.figure()
    plt.subplot(2,1,1)
    x = np.append(deb2[j[0]:j[1],np.newaxis],deb2[j[0]:j[1],np.newaxis],axis=1)
    y = np.append(dfsb2[j[0]:j[1],np.newaxis],dfsb2[j[0]:j[1],np.newaxis],axis=1)
    x[:,0] = 0
    y[:,0] = 0
    #plt.plot(np.transpose(x),np.transpose(y),color='r',alpha=0.5)
    
    x = np.append(deb2[j[1]:j[2],np.newaxis],deb2[j[1]:j[2],np.newaxis],axis=1)
    y = np.append(dfsb2[j[1]:j[2],np.newaxis],dfsb2[j[1]:j[2],np.newaxis],axis=1)
    x[:,0] = 0
    y[:,0] = 0
    #plt.plot(np.transpose(x),np.transpose(y),color='b',alpha=0.5)
    #plt.plot(x[:,1],y[:,1],color='r')
    
    plt.plot(deb2[j[0]:j[1]],dfsb2[j[0]:j[1]],color='r')
    plt.plot(deb2[j[1]-1:j[2]],dfsb2[j[1]-1:j[2]],color='b')
    #plt.xlim([-0.5,0.5])
    #plt.ylim([-0.5,0.5])
    g = plt.gca()
    g.set_aspect('equal')
    plt.subplot(2,1,2)
    plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
    plt.plot([-np.pi,0],[0,np.pi],color='k',linestyle='--')
    plt.plot([0,np.pi],[-np.pi,0],color='k',linestyle='--')
    plt.plot(phase_eb[j[0]:j[1]],phase_fsb[j[0]:j[1]],color='r')
    plt.plot(phase_eb[j[1]-1:j[2]],phase_fsb[j[1]-1:j[2]],color='b')
#%% Phase nulled bump comparison
savedir = r'Y:\Data\FCI\FCI_summaries\FC2_hDeltaC_comparison'
plt.close('all')
bins=5
x = np.linspace(0,1,16)
plotdata_hdc = np.zeros((16,bins*2,2,len(datadirs_hdc)))
for i in range(len(datadirs_hdc)):   
    offset = 0
    cxa = all_flies_hdc[str(i)]
    plotdata_hdc[:,:,:,i] = cxa.phase_nulled_jump(bins=bins,fsb_names=['eb','fsb_upper'],walk='All')
    
plotdata_fc2 = np.zeros((16,bins*2,2,len(datadirs_fc2)))
for i in range(len(datadirs_fc2)): 
    offset = 0
    cxa = all_flies_fc2[str(i)]
    plotdata_fc2[:,:,:,i] = cxa.phase_nulled_jump(bins=bins,fsb_names=['eb','fsb_upper'],walk='All')
    
fig  = plt.figure()    
mplotdata_hdc = np.mean(plotdata_hdc,axis=3)
mplotdata_fc2 = np.mean(plotdata_fc2,axis=3)
#Exit
# hDeltaC
xticks = np.array([0.5])
plt.plot(x,plotdata_hdc[:,bins-1,1,:],color=[0.4,0,1],alpha=0.2)
plt.plot(x,mplotdata_hdc[:,bins-1,1],color=[0.4,0,1],linewidth=2)
offset= 1.1 
xticks = np.append(xticks,offset+0.5)
# FC2
plt.plot(x+offset,plotdata_fc2[:,bins-1,1,:],color='b',alpha=0.2)
plt.plot(x+offset,mplotdata_fc2[:,bins-1,1],color='b',linewidth=2)
offset+= 1.1 
xticks = np.append(xticks,offset+0.5)
# EPG FC2
plt.plot(x+offset,plotdata_hdc[:,bins-1,0,:],color='k',alpha=0.2)
plt.plot(x+offset,mplotdata_hdc[:,bins-1,0],color='k',linewidth=2)

offset+= 1.1 
xticks = np.append(xticks,offset+0.5)
# EPG FC2
plt.plot(x+offset,plotdata_fc2[:,bins-1,0,:],color='k',alpha=0.2)
plt.plot(x+offset,mplotdata_fc2[:,bins-1,0],color='k',linewidth=2)

# Return
offset +=1.6
xticks = np.append(xticks,offset+0.5)
# hDeltaC
plt.plot(x+offset,plotdata_hdc[:,-1,1,:],color=[0.4,0,1],alpha=0.2)
plt.plot(x+offset,mplotdata_hdc[:,-1,1],color=[0.4,0,1],linewidth=2)
offset+= 1.1 
xticks = np.append(xticks,offset+0.5)
# FC2
plt.plot(x+offset,plotdata_fc2[:,-1,1,:],color='b',alpha=0.2)
plt.plot(x+offset,mplotdata_fc2[:,-1,1],color='b',linewidth=2)
offset+= 1.1 
xticks = np.append(xticks,offset+0.5)
# EPG FC2
plt.plot(x+offset,plotdata_hdc[:,-1,0,:],color='k',alpha=0.2)
plt.plot(x+offset,mplotdata_hdc[:,-1,0],color='k',linewidth=2)

offset+= 1.1 
xticks = np.append(xticks,offset+0.5)
# EPG FC2
plt.plot(x+offset,plotdata_fc2[:,-1,0,:],color='k',alpha=0.2)
plt.plot(x+offset,mplotdata_fc2[:,-1,0],color='k',linewidth=2)

plt.xticks(xticks,labels=['hDeltaC','FC2','EPG (hDeltaC)','EPG (FC2)','hDeltaC','FC2','EPG (hDeltaC)','EPG (FC2)'])
plt.ylabel('dF/F0')
fig.set_figwidth(14.39)

plt.savefig(os.path.join(savedir,'PhaseNullComparison.pdf'))


#%% hDeltaC and FC2 phase jump compare
plt.close('all')
for di,f in enumerate(all_flies_hdc):
    cxa = all_flies_hdc[f]
    
    plt.figure(1)
    cxa.mean_jump_arrows_cond(xoffset=di*30,asep=10,colourplot='None',cond='None',fsb_names=['fsb_upper'])
    
    plt.figure(2)
    cxa.mean_jump_arrows_cond(xoffset=di*30,asep=10,colourplot='wmean',cond='None',fsb_names=['fsb_upper'])

for di2,f in enumerate(all_flies_fc2):
    cxa = all_flies_fc2[f]
    
    plt.figure(1)
    cxa.mean_jump_arrows_cond(xoffset=di2*30+di*30+70,asep=10,colourplot='None',cond='None',fsb_names=['fsb_upper'])
    
    plt.figure(2)
    cxa.mean_jump_arrows_cond(xoffset=di2*30+di*30+70,asep=10,colourplot='wmean',cond='None',fsb_names=['fsb_upper'])


savedir = r'Y:\Data\FCI\FCI_summaries\FC2_hDeltaC_comparison'
plt.figure(1)
plt.savefig(os.path.join(savedir,'hDC_FC2_JumpCompNorm.pdf'))

plt.figure(2)
plt.savefig(os.path.join(savedir,'hDC_FC2_JumpCompColoured.pdf'))

#%% Entry versus exit angle phases

plt.close('all')
fig = plt.figure(2)
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
amp_all = np.array([])
phase_all = np.array([])
for f in all_flies_hdc:
    cxa = all_flies_hdc[f]
    jumps = cxa.get_jumps()
    phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()*cxa.side
    amp = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
    
    #amp = cxa.pdat['amp_fsb_upper']
    #amp = np.max(cxa.pdat['wedges_fsb_upper'],axis=1)-np.min(cxa.pdat['wedges_fsb_upper'],axis=1)
    amp = amp/np.std(amp)
    #amp = amp-np.min(amp)
    
    heading= cxa.ft2['ft_heading'].to_numpy()*cxa.side
    for j in jumps:
        ret_p = stats.circmean(phase[j[2]-5:j[2]],high=np.pi,low=-np.pi)
        head_ent = stats.circmean(heading[j[2]-5:j[2]],high=np.pi,low=-np.pi)
        head_ex = stats.circmean(heading[j[1]:j[1]+5],high=np.pi,low=-np.pi)
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.scatter(head_ent,ret_p,color='k',s=5)
        
        plt.subplot(1,2,1)
        plt.scatter(head_ex,ret_p,color='r',s=5)
        plt.xlim([-np.pi,np.pi])
        plt.ylim([-np.pi,np.pi])
        plt.figure(2)
        ax.scatter(phase[j[1]:j[2]],amp[j[1]:j[2]],color='k',s=2,alpha=0.1)
        amp_all = np.append(amp_all,amp[j[1]:j[2]])
        phase_all = np.append(phase_all,phase[j[1]:j[2]])
df = pd.DataFrame({'amp':amp_all,'phase':phase_all})
mean_dat1 = df.groupby(pd.cut(df.phase,np.linspace(-np.pi,np.pi,9))).amp.mean()
bins = np.linspace(-np.pi,np.pi,9)
bins = bins[1:] - np.mean(np.diff(bins))/2
ax.scatter(bins,mean_dat1,color='r')
ax.plot(bins,mean_dat1,color='r')
#ax.set_rticks([])  


  
plt.title('hDeltaC')     
fig = plt.figure(3)
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')


amp_all = np.array([])
phase_all = np.array([])   
for f in all_flies_fc2:
    cxa = all_flies_fc2[f]
    jumps = cxa.get_jumps()
    phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()*cxa.side
    amp = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
   # amp = cxa.pdat['amp_fsb_upper']
    #amp = np.max(cxa.pdat['wedges_fsb_upper'],axis=1)-np.min(cxa.pdat['wedges_fsb_upper'],axis=1)
    amp = amp/np.std(amp)
    #amp = amp-np.min(amp)
    #
    heading= cxa.ft2['ft_heading'].to_numpy()*cxa.side
    for j in jumps:
        ret_p = stats.circmean(phase[j[2]-5:j[2]],high=np.pi,low=-np.pi)
        head_ent = stats.circmean(heading[j[2]-5:j[2]],high=np.pi,low=-np.pi)
        head_ex = stats.circmean(heading[j[1]:j[1]+5],high=np.pi,low=-np.pi)
        plt.figure(1)
        plt.subplot(1,2,2)
        plt.scatter(head_ent,ret_p,color='k',s=5)
        
        plt.subplot(1,2,2)
        plt.scatter(head_ex,ret_p,color='r',s=5)
        plt.xlim([-np.pi,np.pi])
        plt.ylim([-np.pi,np.pi])
        plt.figure(3)
        ax.scatter(phase[j[1]:j[2]],amp[j[1]:j[2]],color='k',s=2,alpha=0.1)
        amp_all = np.append(amp_all,amp[j[1]:j[2]])
        phase_all = np.append(phase_all,phase[j[1]:j[2]])
        
df = pd.DataFrame({'amp':amp_all,'phase':phase_all})
mean_dat2 = df.groupby(pd.cut(df.phase,np.linspace(-np.pi,np.pi,9))).amp.mean()
ax.scatter(bins,mean_dat2,color='r')
ax.plot(bins,mean_dat2,color='r')
plt.title('FC2')
#ax.set_rticks([])    

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
ax.scatter(bins,mean_dat1,color='r')
ax.plot(bins,mean_dat1,color='r')
ax.scatter(bins,mean_dat2,color='b')
ax.plot(bins,mean_dat2,color='b')
#%% Disappearing plumes - look at phase persistence - Check flies
for f in all_flies_hdc:
    cxa = all_flies_hdc[f]
    cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),cxa.pdat['amp_fsb_upper'],a_sep= 5)
    plt.title('hDeltaC ' + f)
    
for f in all_flies_fc2:
    cxa = all_flies_fc2[f]
    cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),cxa.pdat['amp_fsb_upper'],a_sep= 5)
    plt.title('FC2 ' + f)
#%% Disappearing animals
plt.close('all')
hDc_disappear = [3,5]
fc2_disappear = [6]

for f in hDc_disappear:
    cxa = all_flies_hdc[str(f)]
    entries,exits = cxa.get_entries_exits()
    x = cxa.ft2['ft_posx'].to_numpy()
    y = cxa.ft2['ft_posy'].to_numpy()
    x,y = cxa.fictrac_repair(x,y)
    dx = np.arange(entries[-1],len(x))
    
    
    fc2_phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    epg_phase = cxa.pdat['offset_eb_phase'].to_numpy()
    ins = cxa.ft2['instrip'].to_numpy()
    tx = x[dx]
    ty = y[dx]
    tins = ins[dx]
    tfc2 = fc2_phase[dx]
    tepg = epg_phase[dx]
    pdx = np.arange(0,len(tx),5)
    
    plt.figure()
    plt.scatter(tx[tins>0],ty[tins>0],color=[0.7,0.7,0.7])
    plt.plot(x[dx],y[dx],color='k')
    uplt.plot_arrows(tx[pdx],ty[pdx],tfc2[pdx],10,color='b')
    g = plt.gca()
    g.set_aspect('equal', adjustable='box')
    plt.show()
    
    
    
    plt.figure()
    t = dx-dx[0]
    t = t/10
    plt.scatter(t,tfc2,color='b',s=5)
    plt.scatter(t,tepg,color='k',s=5)
    poff = exits[-1]-entries[-1]
    poff = poff/10
    plt.fill([0,poff,poff,0],[-np.pi,-np.pi,np.pi,np.pi],color=[0.7,0.7,0.7],zorder=-1)
    side = -np.sign(x[exits[-1]]-x[entries[-1]])
    plt.plot([0,len(tx)/10],[np.pi/2*side,np.pi/2*side],color='r',linestyle='--')
    plt.plot([0,len(tx)/10],[0,0],color='k',linestyle='--')
    plt.xlim([0,60])
    plt.xlabel('time (s)')
    plt.ylabel('phase (rad)')
    plt.title('hDeltaC '+ str(f))
    
for f in fc2_disappear:
    cxa = all_flies_fc2[str(f)]
    entries,exits = cxa.get_entries_exits()
    x = cxa.ft2['ft_posx'].to_numpy()
    y = cxa.ft2['ft_posy'].to_numpy()
    x,y = cxa.fictrac_repair(x,y)
    dx = np.arange(entries[-1],len(x))
    
    
    fc2_phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    epg_phase = cxa.pdat['offset_eb_phase'].to_numpy()
    ins = cxa.ft2['instrip'].to_numpy()
    tx = x[dx]
    ty = y[dx]
    tins = ins[dx]
    tfc2 = fc2_phase[dx]
    tepg = epg_phase[dx]
    pdx = np.arange(0,len(tx),5)
    
    plt.figure()
    plt.scatter(tx[tins>0],ty[tins>0],color=[0.7,0.7,0.7])
    plt.plot(x[dx],y[dx],color='k')
    uplt.plot_arrows(tx[pdx],ty[pdx],tfc2[pdx],10,color='b')
    g = plt.gca()
    g.set_aspect('equal', adjustable='box')
    plt.show()
    
    
    
    plt.figure()
    t = dx-dx[0]
    t = t/10
    plt.scatter(t,tfc2,color='b',s=5)
    plt.scatter(t,tepg,color='k',s=5)
    poff = exits[-1]-entries[-1]
    poff = poff/10
    plt.fill([0,poff,poff,0],[-np.pi,-np.pi,np.pi,np.pi],color=[0.7,0.7,0.7],zorder=-1)
    side = -np.sign(x[exits[-1]]-x[entries[-1]])
    plt.plot([0,len(tx)/10],[np.pi/2*side,np.pi/2*side],color='r',linestyle='--')
    plt.plot([0,len(tx)/10],[0,0],color='k',linestyle='--')
    plt.xlim([0,60])
    
    plt.xlim([0,60])
    plt.xlabel('time (s)')
    plt.ylabel('phase (rad)')
    plt.title('FC2 '+ str(f))
#%% ET paper scripts


#%%
savedir = r'Y:\Data\FCI\FCI_summaries\FC2_hDeltaC_comparison'
colours2 = np.array([[106,207,246],[237,30,36],[168,170,173],[6,149,207]])/255

timecourse = 15
timecourse_inside= 3

# 2. Realtime
plt.close('all')
for i, d in enumerate(etp_hdc):
    
    etp = etp_hdc[d]
    p1 = etp.phase_time(timecourse)
    p2 = etp.phase_time_inside(timecourse_inside)
    p = np.append(p1,p2,axis=0)
    p[np.abs(p)==0.0] = np.nan
    pmean = stats.circmean(p,high=np.pi, low=-np.pi,axis=2,nan_policy='omit') #%need to do circ mean
    t = np.arange(0,timecourse+timecourse_inside,0.1)
    #plt.plot(t,p[:,0,:],color=colours[0,:],alpha=0.1,linestyle=':')
    #plt.plot(t,p[:,1,:],color=colours[1,:],alpha=0.1,linestyle=':')
    tt = np.tile(t,(np.shape(p)[2],1))
    #plt.scatter(tt,p[:,0,:],alpha=0.4,s=5,color=colours[0,:])
    #plt.scatter(tt,p[:,1,:],alpha=0.4,s=5,color=colours[1,:])
    #plt.plot(t,pmean[:,0],color=colours[0,:],alpha=1)
    #plt.plot(t,pmean[:,1],color=colours[1,:],alpha=1)
    if i==0:
        pltmean = np.zeros((len(pmean),4,len(etp_hdc)))
    pltmean[:,:,i] = 180*pmean/np.pi


colours = np.array([[81,156,204],[84,39,143],[0,0,0],[6,149,207]])/255
colours = np.array([[0,0,0],[81,156,204]])/255
plt.figure()
plt.subplot(1,2,1)
pm = stats.circmean(pltmean,high=180,low=-180,axis=2,nan_policy='omit')
zorder = [1,3,0]
for i in range(2):
    
    #plt.plot(pltmean[:,i,:],t,color = colours[i,:],alpha=0.3,zorder=zorder[i])
    for i2 in range(pltmean.shape[2]):
        plt.scatter(pltmean[:,i,i2],t,color = colours[i,:],alpha=0.4,zorder=zorder[i],s=4)
    plt.plot(pm[:,i],t,color=colours[i,:],zorder=zorder[i])
 
    
    
plt.plot([-180,180],[timecourse,timecourse],color='k',linestyle='--')
plt.plot([0,0],[0,timecourse+timecourse_inside],color='k',linestyle='--')
plt.plot([-90,-90],[0,timecourse],color=colours2[1,:],linestyle='--')
plt.xlim([-180,180])
plt.xlabel('angle (deg)')
plt.ylabel('time (s)')
plt.yticks(np.arange(0,17.6,2.5),labels=[-15,-12.5,-10,-7.5,-5,-2.5,0,2.5]);
plt.ylim([0,timecourse+timecourse_inside+0.2])
plt.xticks([-180,0,180])
plt.title('hDeltaC')
plt.show()

for i, d in enumerate(etp_fc2):
    
    etp = etp_fc2[d]
    p1 = etp.phase_time(timecourse)
    p2 = etp.phase_time_inside(timecourse_inside)
    p = np.append(p1,p2,axis=0)
    p[np.abs(p)==0.0] = np.nan
    pmean = stats.circmean(p,high=np.pi, low=-np.pi,axis=2,nan_policy='omit') #%need to do circ mean
    t = np.arange(0,timecourse+timecourse_inside,0.1)
    #plt.plot(t,p[:,0,:],color=colours[0,:],alpha=0.1,linestyle=':')
    #plt.plot(t,p[:,1,:],color=colours[1,:],alpha=0.1,linestyle=':')
    tt = np.tile(t,(np.shape(p)[2],1))
    #plt.scatter(tt,p[:,0,:],alpha=0.4,s=5,color=colours[0,:])
    #plt.scatter(tt,p[:,1,:],alpha=0.4,s=5,color=colours[1,:])
    #plt.plot(t,pmean[:,0],color=colours[0,:],alpha=1)
    #plt.plot(t,pmean[:,1],color=colours[1,:],alpha=1)
    if i==0:
        pltmean = np.zeros((len(pmean),4,len(etp_hdc)))
    
    pltmean[:,:,i] = 180*pmean/np.pi


colours = np.array([[81,156,204],[84,39,143],[0,0,0],[6,149,207]])/255
colours = np.array([[0,0,0],[81,156,204]])/255

plt.subplot(1,2,2)
pm = stats.circmean(pltmean,high=180,low=-180,axis=2,nan_policy='omit')
zorder = [1,3,0]
for i in range(2):
    #plt.plot(pltmean[:,i,:],t,color = colours[i,:],alpha=0.3,zorder=zorder[i])
    for i2 in range(pltmean.shape[2]):
        plt.scatter(pltmean[:,i,i2],t,color = colours[i,:],alpha=0.4,zorder=zorder[i],s=4)
    plt.plot(pm[:,i],t,color=colours[i,:],zorder=zorder[i])
 
    
    
plt.plot([-180,180],[timecourse,timecourse],color='k',linestyle='--')
plt.plot([0,0],[0,timecourse+timecourse_inside],color='k',linestyle='--')
plt.plot([-90,-90],[0,timecourse],color=colours2[1,:],linestyle='--')
plt.xlim([-180,180])
plt.xlabel('angle (deg)')
plt.ylabel('time (s)')

plt.ylim([0,timecourse+timecourse_inside+0.2])
plt.xticks([-180,0,180])
plt.title('FC2')
plt.yticks(np.arange(0,17.6,2.5),labels=[-15,-12.5,-10,-7.5,-5,-2.5,0,2.5]);
plt.show()

plt.savefig(os.path.join(savedir,'FC2_hDC_timecourse_comp.pdf'))




#%% Scrap paper
plt.close('all')
for f in all_flies_fc2:
    cxa = all_flies_fc2[f]
    plt.figure()
    pdiff = ug.circ_subtract(cxa.pdat['phase_eb'],cxa.ft2['ft_heading'])
    x = np.arange(0,len(pdiff))
    plt.scatter(x,pdiff,s=5,c=cxa.ft2['ft_heading'],cmap='coolwarm')
    smth = ug.savgol_circ(pdiff,60,3)
   # plt.plot(x,smth,color='k')
   # plt.plot(x,cxa.pdat['offset'].to_numpy(),color='r')
    #plt.scatter(cxa.pdat['phase_eb'][:5000],cxa.ft2['ft_heading'][:5000],color='k',s=3)
for f in all_flies_hdc:
    cxa = all_flies_hdc[f]
    plt.figure()
    pdiff = ug.circ_subtract(cxa.pdat['phase_eb'],cxa.ft2['ft_heading'])
    x = np.arange(0,len(pdiff))
    plt.scatter(x,pdiff,s=5,c=cxa.ft2['ft_heading'],cmap='coolwarm')
    smth = ug.savgol_circ(pdiff,60,3)
    #plt.plot(x,smth,color='k')
    #plt.plot(x,cxa.pdat['offset'].to_numpy(),color='r')
    #plt.scatter(cxa.pdat['phase_eb'][:5000],cxa.ft2['ft_heading'][:5000],color='k',s=3)


















