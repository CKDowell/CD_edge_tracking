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
#%% Load up data
datadirs_hdc = [
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250411\f1\Trial1",# Phase recording is not the best - strong pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial3",# Not many jumps, weak pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f2\Trial2", # Strong pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial2",# Strong pointer, just like FC2
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial1", # Points away ******
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial3", # Strong pointer
                r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial2",# Strong pointer
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
for i,datadir in enumerate(datadirs_fc2):
    print(datadir)
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    all_flies_fc2pam.update({str(i):cxa})
    etp = ET_paper(datadir)
    etp_fc2pam.update({str(i):etp})

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























