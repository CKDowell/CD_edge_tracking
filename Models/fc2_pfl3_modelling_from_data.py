# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:42:56 2026

@author: dowel

FC2 PFL3 modelling


"""

#%% Load data
from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper
from Utilities.utils_general import utils_general as ug
from Utilities.utils_plotting import uplt
import os
import numpy as np
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
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260425\f2\Trial3', # 3 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260401\f1\Trial2', # Some nice tracking 3 jumps
        r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260429\f2\Trial1',# 17 jumps
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
        else:
            etp = ET_paper(datadir)
        all_flies[group_name][i] = etp
#%% Predict PFL3
from CD_edge_tracking.Models.neuro2pfl3 import pfl3_model
import matplotlib.pyplot as plt
dchoice = ['FC2','hDeltaC_FC2_EPG']

for d in dchoice:
    tdata = all_flies[d]
    for i in tdata:
        print(i)
        etp = tdata[i]
        mdl = pfl3_model()
        if d=='FC2':
            phase_eb = etp.cxa.pdat['phase_eb']
            phase_goal = etp.cxa.pdat['phase_fsb_upper']
            fsb  = etp.cxa.pdat['wedges_fsb_upper']
            ebw = etp.cxa.pdat['wedges_eb']
        else:
            phase_eb = etp.cxa.pdat['phase_eb_ch1']
            phase_goal = etp.cxa.pdat['phase_fsb1_ch1']
            fsb  = etp.cxa.pdat['wedges_fsb1_ch1']
            ebw = etp.cxa.pdat['wedges_eb_ch1']
        L,R,turns = mdl.model_pfl3_phase(phase_eb,phase_goal,goal_weight=2,eb_function='cosine',goal_function='cosine')
        
        
        tdata[i].pfl3_model = {'all_phase': np.column_stack((L,R,turns))}
        
        L,R,turns2 =  mdl.model_pfl3_fsb_data(phase_eb,fsb,goal_weight=5,eb_function='cosine',goal_function='cosine')
        tdata[i].pfl3_model.update({'phase_fsb_col': np.column_stack((L,R,turns2))})
        
        L,R,turns3 = mdl.model_pfl3_all_data(ebw,fsb,goal_weight=10,d7weight=3)
        tdata[i].pfl3_model.update({'all_data': np.column_stack((L,R,turns3))})
        
        if d=='hDeltaC_FC2_EPG':
            #Model as if it was getting pure hDeltaC
            phase_goal = etp.cxa.pdat['phase_fsb2_ch2']
            fsb  = etp.cxa.pdat['wedges_fsb2_ch2']
            L,R,turns = mdl.model_pfl3_phase(phase_eb,phase_goal,goal_weight=2,eb_function='cosine',goal_function='cosine')
            tdata[i].pfl3_model.update( {'all_phase_hdc': np.column_stack((L,R,turns))})
            
            L,R,turns2 =  mdl.model_pfl3_fsb_data(phase_eb,fsb,goal_weight=5,eb_function='cosine',goal_function='cosine')
            tdata[i].pfl3_model.update({'phase_fsb_col_hdc': np.column_stack((L,R,turns2))})
            
            L,R,turns3 = mdl.model_pfl3_all_data(ebw,fsb,goal_weight=10,d7weight=3)
            tdata[i].pfl3_model.update({'all_data_hdc': np.column_stack((L,R,turns3))})
        
    all_flies[d] = tdata
#%% 
"""
To do:
    1. Inspect trajectories
    2. Extract turns and see how far in advance the PFL3 signal can predict them
    3. Look at stop and starts to see how the signal may change during these time periods
    
"""
dchoice = 'FC2'
fly =3
etp = all_flies[dchoice][fly]
turns2 = etp.pfl3_model['phase_fsb_col'][:,2].copy()
tstart = 60*4.2 + 10
tend = 60*4.2 + 60
etp.example_trajectory(tstart,tend,boxes=[],heat=True,col=turns2.copy(),cmap='coolwarm',cmin=-.25,cmax=.25)
colours = uplt.columnar_colours()
colours = colours[[1,2]]
colours[0,:] = 0

etp.example_trajectory_arrows(tstart,tend,colours,samp_period=2,boxes=[],heat=True,col=turns2.copy(),cmap='coolwarm_r',cmin=-.25,cmax=.25)

savedir =  r'Y:\Presentations\2026\05_ForVanessaAndLarry'
plt.savefig(os.path.join(savedir,'Example_PFL3_Activity.pdf'))


if dchoice=='FC2':
    etp.cxa.plot_traj_arrow_heat(['fsb_upper'],turns2,cmin=-.25,cmax=.25,a_sep=5)
else:
    etp.cxa.plot_traj_arrow_heat(['fsb1_ch1'],turns2,cmin=-.25,cmax=.25,a_sep=5)




plt.figure()
plt.hist(turns,100)
turns2 = etp.pfl3_model['phase_fsb_col'][:,2] 
plt.hist(turns2,100)
turns3 = etp.pfl3_model['all_phase'][:,2] 
#plt.hist(turns3,100)



#%% Inspect data
from Utilities.utils_plotting import uplt
colours = uplt.columnar_colours()
dchoice = 'hDeltaC_FC2_EPG'
fly =4
etp = all_flies[dchoice][fly]
u = ug()
jumps = etp.cxa.get_jumps()
turns2 = etp.pfl3_model['phase_fsb_col'][:,2] 
turns3 = etp.pfl3_model['all_phase'][:,2] 
cxa = etp.cxa
dx,dy,dd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'].to_numpy())
heading = cxa.ft2['ft_heading'].to_numpy()
dd = np.append(0,dd)
dd = dd/np.std(dd)
offset = 0
plt.figure()
plt.title(etp.cxa.side)
turns2 = turns2/np.std(turns2)
turns = turns/np.std(turns)
tstd2 = np.std(etp.pfl3_model['phase_fsb_col'][:,2])
yarray = np.array([-np.pi,-np.pi,np.pi,np.pi])
for i,j in enumerate(jumps):
    jdx = np.arange(j[0],j[2])
    jst = j[1]
    tt = (jdx-jst)/10 
    t_turn = turns2[jdx]
    #t_turno = turns[jdx]
    t_vel = dd[jdx]
    t_h = heading[jdx]
    #plt.plot(tt,t_vel+offset,color='k')
    plt.fill([tt[0],0,0,tt[0]],yarray+offset,color=[.8,.8,.8],zorder=-1)
    plt.scatter(tt,t_h+offset,s=5,color='k')
    
    
    
    plt.plot(tt,t_turn+offset,color=colours[2,:])
    if dchoice=='hDeltaC_FC2_EPG':
        turns3 = etp.pfl3_model['phase_fsb_col_hdc'][jdx,2] /tstd2
        plt.plot(tt,turns3+offset,color=colours[1,:])
    #plt.plot(tt,t_turno+offset,color=[1,.6,.6])
    
    
    
    plt.plot(tt,tt*0+offset,color='k')
    offset = offset-7
    

#%% 
plt.close('all')
from scipy import stats
dchoice = 'FC2'
tdata = all_flies[dchoice]
xoffset = 0
xplume = np.array([-10,0,0,-10])
psamples = np.linspace(0,199,8).astype(int)
colours = uplt.columnar_colours()[[0,2],:]
x = np.append(np.arange(0,50,.5),np.linspace(50,200,100))-50
colours[0,:] = 0
grandmean = np.zeros((200,len(tdata)))
for t in tdata:
    etp = tdata[t]
    trajs,turns = etp.trajectory_mean_variable(etp.pfl3_model['phase_fsb_col'][:,2],side_mult=True)
    phase,_,amp = etp.trajectory_mean(regions=['eb','fsb_upper'])
    phase = stats.circmean(phase,axis=2,low=-np.pi,high=np.pi)
    amean = np.mean(amp,axis=3)
    tmean = np.mean(trajs,axis=2)
    turnmean = np.mean(turns,axis=2)
    grandmean[:,t] = turnmean.squeeze()
    plt.figure(101)
    yplume = np.array([0,0,np.min(tmean[:,1]),np.min(tmean[:,1])])
    plt.fill(xplume+xoffset,yplume,color=[.8,.8,.8])
    yplume = np.array([0,0,np.max(tmean[:,1]),np.max(tmean[:,1])])
    plt.fill(xplume+xoffset-3,yplume,color=[.8,.8,.8])
    
    uplt.coloured_line_simple(tmean[:,0]+xoffset,tmean[:,1], turnmean, 'coolwarm_r', -.25, .25)
    
    for p in psamples:
        for i in range(2):
            scale = amean[p,0,i]*30
            tx = np.array([tmean[p,0],tmean[p,0]+np.sin(phase[p,i])*scale])
            ty = np.array([tmean[p,1],tmean[p,1]+np.cos(phase[p,i])*scale])
            plt.plot(tx+xoffset,ty,color=colours[i,:])
    xoffset = xoffset+20
    
    
    plt.figure(102)
    #plt.plot(x,turnmean,color=colours[1,:])
    uplt.coloured_line_simple(x,turnmean, turnmean, 'coolwarm_r', -.25, .25)
    plt.figure(103)
    plt.plot(x,turnmean-turnmean[100],color=colours[1,:])

plt.figure(101)
plt.gca().set_aspect('equal')

plt.figure(102)
gm = np.mean(grandmean,axis=1)
#uplt.coloured_line_simple(x, gm, gm, 'coolwarm', -.25, .25,linewidth=3)
plt.plot(x, gm,color='k',linewidth=3)
plt.plot([-50,150],[0,0],color='k')
plt.ylim([-.4,.4])
plt.fill([-50,0,0,-50],[-.4,-.4,.4,.4],color=[.8,.8,.8],zorder=-1)
plt.ylabel('PFL3 turn amplitude')
plt.xlabel('Pseudotime')

#%% Stop start dynamics - jumps


from Utilities.utils_plotting import uplt
colours = uplt.columnar_colours()
dchoice = 'FC2'
fly =4
flycolours = uplt.defined_cmap('hsv',7) 
for fly in range(7):
    etp = all_flies[dchoice][fly]
    u = ug()
    jumps = etp.cxa.get_jumps()
    turns2 = etp.pfl3_model['phase_fsb_col'][:,2] 
    turns3 = etp.pfl3_model['all_phase'][:,2] 
    cxa = etp.cxa
    dx,dy,dd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'].to_numpy())

    allstps = dd<.5
    bstart,bsize = ug.find_blocks(allstps)
    minsize = 10 
    bstart = bstart[bsize>=minsize]
    bsize = bsize[bsize>=minsize]
    heading = cxa.ft2['ft_heading'].to_numpy()
    plt.figure(fly)
    for i,j in enumerate(jumps):
        jdx = np.arange(j[1],j[2])
        tbs = np.logical_and(bstart>j[1],bstart<j[2])
        theseblocks = bstart[tbs]
        thesesizes = bsize[tbs]
        for t in range(sum(tbs)):
            tbdx = np.arange(theseblocks[t],theseblocks[t]+thesesizes[t])
            adx = np.arange(theseblocks[t]+thesesizes[t]+2,theseblocks[t]+thesesizes[t]+5+2)
            tmn = np.mean(turns2[tbdx[-5:]])*-etp.cxa.side
            hmn = stats.circmean(heading[tbdx],low=-np.pi,high=np.pi)
            hmn2 = stats.circmean(heading[adx],low=-np.pi,high=np.pi)
            dh = ug.circ_subtract(hmn2,hmn)*-etp.cxa.side
            #if np.abs(dh)>.1:
                #plt.scatter(tmn,dh,color=flycolours[fly,:],s=10)
            plt.scatter(tmn,dh,color='r',s=10)
    plt.xlim([-.5,.5])
    plt.plot([0,0],[-2,2],color='k')
    plt.plot([-.4,.4],[0,0],color='k')

#%% Stop start dynamics - all data

for fly in range(7):
    etp = all_flies[dchoice][fly]
    u = ug()
    jumps = etp.cxa.get_jumps()
    turns2 = etp.pfl3_model['phase_fsb_col'][:,2] 
    turns3 = etp.pfl3_model['all_phase'][:,2] 
    cxa = etp.cxa
    dx,dy,dd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'].to_numpy())

    allstps = dd<.5
    bstart,bsize = ug.find_blocks(allstps)
    minsize = 10 
    bstart = bstart[bsize>=minsize]
    bsize = bsize[bsize>=minsize]
    heading = cxa.ft2['ft_heading'].to_numpy()
    
    plt.figure(fly)
    for ib,b in enumerate(bstart):
        tbdx = np.arange(b,b+bsize[ib])
        adx = np.arange(b+bsize[ib]+2,b+bsize[ib]+5+2)
        if np.max(adx)>len(turns2):
            continue
        tmn = np.mean(turns2[tbdx[-5:]])*-etp.cxa.side
        hmn = stats.circmean(heading[tbdx],low=-np.pi,high=np.pi)
        hmn2 = stats.circmean(heading[adx],low=-np.pi,high=np.pi)
        dh = ug.circ_subtract(hmn2,hmn)*-etp.cxa.side
        if np.abs(dh)>.1:
            #plt.scatter(tmn,dh,color=flycolours[fly,:],s=10)
            plt.scatter(tmn,dh,color='k',s=5)

    
    plt.xlim([-.5,.5])
    plt.plot([0,0],[-2,2],color='k')
    plt.plot([-.4,.4],[0,0],color='k')







































