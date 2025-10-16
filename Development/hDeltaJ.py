# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:16:41 2024

@author: dowel
"""
from analysis_funs.regression import fci_regmodel

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
from src.utilities import funcs as fn
from scipy import stats
from Utilities.utils_general import utils_general as ug

plt.rcParams['pdf.fonttype'] = 42 
#%% Image registraion

for i in [1,2,3,4,5]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\hDeltaJ\251011\f1\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    #ex.register_all_images(overwrite=True)
    #ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
#%% Basic data processing
experiment_dirs = [
 
                   #"Y:\\Data\\FCI\\Hedwig\\hDeltaJ\\240529\\f1\\Trial3",
                   #r"Y:\Data\FCI\Hedwig\hDeltaJ\250926\f1\Trial3" # not quite making all jumps, simply anti heading and inhibited in plume
                   
                   r"Y:\Data\FCI\Hedwig\hDeltaJ\251011\f1\Trial1", # Nice trial backwards pointing vector towards plume
                   r"Y:\Data\FCI\Hedwig\hDeltaJ\251011\f1\Trial2", # Again, nice backwards vector towards plume
                   r"Y:\Data\FCI\Hedwig\hDeltaJ\251011\f1\Trial3", # Not amazing tracker, crossed over plumes
                   r"Y:\Data\FCI\Hedwig\hDeltaJ\251011\f1\Trial4", # ACV pulses
                   r"Y:\Data\FCI\Hedwig\hDeltaJ\251011\f1\Trial5" # Oct pulses
                   ]
for e in experiment_dirs:
    datadir =os.path.join(e)
    print(e)
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cx = CX(name,['fsb_lower','fsb_upper','eb'],datadir)
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois()
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()
    pv2, ft, ft2, ix = cx.load_postprocessing()

    try :
        cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'])
    except:
        cxa = CX_a(datadir,regions=['eb','fsb'])
    
    cxa.save_phases()
    
#%% Data exploration
datadir = r"Y:\Data\FCI\Hedwig\hDeltaJ\240529\\f1\Trial3"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
plt.close('all')
cxa.simple_raw_plot(plotphase=True,regions = ['fsb_upper','fsb_lower'])
#cxa.simple_raw_plot(plotphase=True)
#%% 
cxa.point2point_heat(3500,4500,toffset=0,arrowpoint=np.array([50,243,500,600,700,800,900]))
cxa.point2point_heat(0,1000,toffset=0,arrowpoint=np.array([50,243,500,600,700,800,900]))

#%% Jump arrows
datadirs = [
    "Y:\\Data\\FCI\\Hedwig\\hDeltaJ\\240529\\f1\\Trial3"]
plt.close('all')
x_offset = 0
plt.figure()
for datadir in datadirs:
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    cxa.save_phases()
    cxa.mean_jump_arrows(x_offset,fsb_names=['fsb_upper'],ascale=100)
    x_offset = x_offset+30


plt.ylim([-40,40])
savedir= 'Y:\\Data\\FCI\\FCI_summaries\\hDeltaJ'
plt.savefig(os.path.join(savedir,'MeanJumps.png'))
plt.savefig(os.path.join(savedir,'MeanJumps.pdf'))
#%%
cxa.mean_jump_arrows(x_offset,fsb_names=['fsb_upper'])
#%% regression for hDj
regchoice = ['odour onset', 'odour offset', 'in odour', 
                             'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
  'angular velocity pos','angular velocity neg','translational vel','ramp down since exit','ramp to entry']

dr2mat = np.zeros((len(datadirs),len(regchoice)))
dr2mat_max = np.zeros((len(datadirs),len(regchoice)))
savedir = "Y:\\Data\\FCI\\FCI_summaries\\hDeltaJ"
angles = np.linspace(-np.pi,np.pi,16)
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
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    dr2mat[ir,:] = (-fc.dR2_mean)*np.sign(fc.coeff_cv[:-1])
    
    y = np.mean(cxa.pdat['wedges_offset_fsb_upper'],axis=1)
    ft2 = cxa.ft2
    pv2 = cxa.pv2
    fc = fci_regmodel(y,ft2,pv2)
    #fc.rebaseline()
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    dr2mat_max[ir,:] = (-fc.dR2_mean)*np.sign(fc.coeff_cv[:-1])
    
    fc.plot_example_flur()
    plt.title('Fly: ' + str(ir) +  ' R2:' +str(fc.cvR2))
    plt.savefig(os.path.join(savedir,'EgFit_' + str(ir)+ '.png'))
    plt.figure()
    plt.title(str(ir))
    fc.plot_flur_w_regressors(['in odour','translational vel'],cacol= 'r')
    plt.savefig(os.path.join(savedir,'Ca_withreg_' + str(ir)+ '.png'))
    
#%% 
plt.figure()
x = np.arange(0,len(regchoice))
plt.plot([0,len(regchoice)],[0,0],linestyle='--',color='k')
plt.plot(x,np.transpose(dr2mat),color='k')

plt.xticks(x,labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.title('PVA amplitude regression')
plt.ylabel('Signed dR2')
plt.savefig(os.path.join(savedir,'Reg_Bump_PVA_dR2.png'))
plt.savefig(os.path.join(savedir,'Reg_Bump_PVA_dR2.pdf'))
plt.figure()
x = np.arange(0,len(regchoice))
plt.plot([0,len(regchoice)],[0,0],linestyle='--',color='k')
plt.plot(x,np.transpose(dr2mat_max),color='k')
plt.xticks(x,labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.title('Mean columns')
plt.ylabel('Signed dR2')
plt.savefig(os.path.join(savedir,'Reg_Bump_Mean_dR2.png'))
plt.savefig(os.path.join(savedir,'Reg_Bump_Mean_dR2.pdf'))
#%%
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
plt.close('all')
mi = -0.4
mxi = 0.1
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

#%% 
datadir=  r"Y:\\Data\\FCI\\Hedwig\\hDeltaJ\\240529\\f1\\Trial3"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
#%%
savedir = r'Y:\Data\FCI\FCI_summaries\hDeltaJ'
cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb')
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_lower_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_lower']/2,axis=1),a_sep= 2)

plt.figure()
t  = np.arange(0,len(cxa.pdat['phase_eb']))/10
plt.scatter(t,cxa.pdat['phase_eb'],color='k',s=2)
plt.scatter(t,cxa.pdat['phase_fsb_upper'],color='b',s=2)
plt.scatter(t,cxa.pdat['phase_fsb_lower'],color='m',s=2)
plt.plot(t,cxa.ft2['instrip']*3,color='r')

plt.figure()
cd = np.abs(ug.circ_subtract(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_eb']))
cd2 = np.abs(ug.circ_subtract(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_fsb_lower']))
plt.plot(t,cd,color='b')
#plt.plot(t,cd2,color='m')
plt.plot(t,cxa.ft2['instrip'],color='r')
plt.figure()
cxa.mean_jump_arrows(fsb_names=['fsb_upper','fsb_lower'],jsize =3,ascale=100)

#%%
plt.figure()
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],linestyle='--',color='r')
plt.plot([-np.pi,0],[0,np.pi],linestyle='--',color='r')
plt.plot([0,np.pi],[-np.pi,0],linestyle='--',color='r')

plt.scatter(cxa.pdat['phase_eb'],cxa.pdat['phase_fsb_upper'],color='k',s=1,alpha=.2)
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.xlabel('EPG phase')
plt.ylabel('FSB upper phase')
plt.savefig(os.path.join(savedir,'FSB_EPG_Phase.png'))

plt.figure()
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],linestyle='--',color='r')
plt.plot([-np.pi,0],[0,np.pi],linestyle='--',color='r')
plt.plot([0,np.pi],[-np.pi,0],linestyle='--',color='r')

plt.scatter(cxa.pdat['phase_eb'][:-20],cxa.pdat['phase_fsb_upper'][20:],color='k',s=1,alpha=.2)
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.xlabel('EPG phase')
plt.ylabel('FSB upper phase')
#plt.savefig(os.path.join(savedir,'FSB_EPG_Phase.png'))


plt.figure()
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],linestyle='--',color='r')
plt.plot([-np.pi,0],[0,np.pi],linestyle='--',color='r')
plt.plot([0,np.pi],[-np.pi,0],linestyle='--',color='r')
plt.scatter(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_fsb_lower'],color='k',s=1,alpha=.2)
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.xlabel('FSB upper phase')
plt.ylabel('FSB lower phase')
plt.savefig(os.path.join(savedir,'FSB_upper_lower_phase.png'))
#%%
x = np.arange(0,len(cxa.pdat['phase_fsb_upper']))/10
plt.scatter(x,cxa.pdat['phase_fsb_upper'],color='b',s=5)
plt.scatter(x,cxa.pdat['phase_eb'],color='k',s=2)
eb180 = ug.circ_subtract(cxa.pdat['phase_eb'],np.pi)

eb180_filt = ug.savgol_circ(eb180,40,3)
plt.scatter(x,eb180,color=[0,0,0],s=5)
plt.scatter(x,eb180_filt,color=[0.2,0.2,0.2],s=5)   
    

#%%  Time lag
eb180_diff = ug.circ_vel(eb180,x,smooth=False,winlength=10)
fsb_diff = ug.circ_vel(cxa.pdat['phase_fsb_upper'],x,smooth=False,winlength=10)
c = sg.correlate(eb180_diff,fsb_diff)
c= c/np.max(c)
lags = sg.correlation_lags(len(eb180_diff),len(fsb_diff))/10
plt.plot(lags,c)
plt.plot([0,0],[0,1],color='k',linestyle='--')
plt.xlim([-5,5])
plt.xticks(np.arange(-5,5))
plt.xlabel('Time lag')
pk = np.argmax(c[:int(len(c)/2-2.5)])
plt.scatter(lags[pk],1)
plt.text(lags[pk]-1,1,str(lags[pk]))
plt.ylabel('Cross correlation')
#%% Stationary during jump returns vs walking
# See compare script

#%% Plot every return start-stop
plt.close('all')
minsize = 3
u = ug()
e_e = cxa.get_entries_exits_like_jumps()
phase_eb = cxa.pdat['phase_eb']
phase_fsb = cxa.pdat['phase_fsb_upper']
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()

tt = cxa.pv2['relative_time'].to_numpy()
dx,dy,dd = u.get_velocity(x,y,tt)
stat = dd<1
blockstart,blocksize = ug.find_blocks(stat,mergeblocks=True,merg_threshold=2)
blockstart = blockstart[blocksize>minsize]
blocksize = blocksize[blocksize>minsize]

stat2 = np.zeros(len(stat),dtype='int')
mv = stat==0
for i,b in enumerate(blockstart):
    bdx = np.arange(b,blocksize[i]+b)
    stat2[bdx ] = 1*i
    if i<len(blockstart)-1:
        bdx2 = np.arange(blocksize[i]+b,blockstart[i+1])
        stat2[bdx2 ] = -1*i


mean_array = np.empty((len(e_e),100,2,2))
mean_array[:] = np.nan
for i,e in enumerate(e_e):
    edx = np.arange(e[1],e[2])
    tstat  = stat2[edx]
    tphase_fsb = phase_fsb[edx]
    tphase_eb = phase_eb[edx]
    stops = np.unique(tstat[tstat>0])
    for i2,s in enumerate(stops):
        r = np.random.randn(1)*.05
        ts = s-min(stops)+r
        x = stats.circmean(tphase_eb[tstat==s],high=np.pi,low=-np.pi)
        y = stats.circmean(tphase_fsb[tstat==s],high=np.pi,low=-np.pi)
        mean_array[i,i2,0,0] = x
        mean_array[i,i2,1,0] = y
       # plt.scatter(ts,x,color='k',s=2)
        plt.scatter(ts,y,color='b',s=2)
        
        x = stats.circmean(tphase_eb[tstat==-(s-1)],high=np.pi,low=-np.pi)
        y = stats.circmean(tphase_fsb[tstat==-(s-1)],high=np.pi,low=-np.pi)
        plt.scatter(ts-0.5,x,color='k',s=2)
        plt.scatter(ts-0.5,y,color='b',s=2)
        mean_array[i,i2,0,1] = x
        mean_array[i,i2,1,1] = y

pltmean = stats.circmean(mean_array,axis=0,nan_policy='omit',high=np.pi,low=-np.pi)
x = np.arange(0,len(pltmean))

plt.scatter(x,pltmean[:,0,0],color='k')
plt.plot(pltmean[:,0,0],color='k')
plt.scatter(x,pltmean[:,1,0],color='b')
plt.plot(pltmean[:,1,0],color='b')

plt.scatter(x-.5,pltmean[:,0,1],color='k',marker='+')
plt.plot(x-.5,pltmean[:,0,1],color='k')
plt.scatter(x-.5,pltmean[:,1,1],color='b',marker='+')
plt.plot(x-.5,pltmean[:,1,1],color='b')

#%%
plt.close('all')

mean_array = np.empty((len(e_e),1000,100))
mean_array[:] = np.nan
for i,e in enumerate(e_e):
    edx = np.arange(e[1],e[2])
    tstat  = stat2[edx]
    tphase_fsb = phase_fsb[edx]
    tphase_eb = phase_eb[edx]
    stops = np.unique(tstat[tstat>0])
    for i2,s in enumerate(stops):
        plt.figure(i2)
        r = np.random.randn(1)*.05
        ts = s-min(stops)+r
        x = tphase_eb[tstat==-(s-1)]
        y = tphase_fsb[tstat==-(s-1)]
        
        t = np.arange(-len(x),0)/10
        
        plt.scatter(t,ug.circ_subtract(y,x),color=[0,0,1],s=5)
        mean_array[i,500-len(x):500,i2] = ug.circ_subtract(y,x)
        
        
        
        x = tphase_eb[tstat==s]
        y = tphase_fsb[tstat==s]
        t = np.arange(0,len(x))/10+np.max(t)
        plt.scatter(t,ug.circ_subtract(y,x),color=[0,0,0.5],s=5)
        mean_array[i,500:500+len(x),i2] = ug.circ_subtract(y,x)
      
        
      
for i in range(3):
    plt.figure(i)
    x = np.arange(-500,500)/10
    pmn = stats.circmean(mean_array,axis=0,nan_policy='omit',high=np.pi,low=-np.pi)
    plt.plot(x[:500],pmn[:500,i],color= [0,0,1])
    plt.plot(x[500:],pmn[500:,i],color=[0,0,0.5])
    plt.plot([0,0],[-np.pi,np.pi],color='r',linestyle='--')
    plt.xlabel('time from movement offset (s)')
    plt.ylabel('FSB-EB phase (rad)')
    plt.ylim([-np.pi,np.pi])
mean_array = np.empty((len(e_e),1000,100))
mean_array[:] = np.nan
for i,e in enumerate(e_e):
    edx = np.arange(e[1],e[2])
    tstat  = stat2[edx]
    tphase_fsb = phase_fsb[edx]
    tphase_eb = phase_eb[edx]
    stops = np.unique(tstat[tstat>0])
    for i2,s in enumerate(stops):
        plt.figure(i2+100)
        r = np.random.randn(1)*.05
        ts = s-min(stops)+r
        x = tphase_eb[tstat==s]
        y = tphase_fsb[tstat==s]
        t = np.arange(-len(x),0)/10
        plt.scatter(t,ug.circ_subtract(y,x),color=[0,0,0.5],s=2) 
        mean_array[i,500-len(x):500,i2] = ug.circ_subtract(y,x)
        t = np.arange(-len(x),0)
        
        x = tphase_eb[tstat==-(s+1)]
        y = tphase_fsb[tstat==-(s+1)]
        t = np.arange(0,len(x))/10+np.max(t)+1
        plt.scatter(t,ug.circ_subtract(y,x),color=[0,0,1],s=2)    
        mean_array[i,500:500+len(x),i2] = ug.circ_subtract(y,x)
               
        
for i in range(3):
    plt.figure(100+i)
    x = np.arange(-500,500)/10
    pmn = stats.circmean(mean_array,axis=0,nan_policy='omit',high=np.pi,low=-np.pi)
    plt.plot(x[:500],pmn[:500,i],color=[0,0,0.5] )
    plt.plot(x[500:],pmn[500:,i],color=[0,0,1])
    plt.plot([0,0],[-np.pi,np.pi],color='r',linestyle='--')
    plt.xlabel('time from movement onset (s)')
    plt.ylabel('FSB-EB phase (rad)')
    plt.ylim([-np.pi,np.pi])