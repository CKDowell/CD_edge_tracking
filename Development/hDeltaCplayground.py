# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:42:13 2023

@author: dowel
"""
#%% 
import numpy as np

import os
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from src.utilities import funcs as fn
from scipy import stats
datadir = "Y:\\Data\\FCI\\AndyData\\hDeltaC_imaging\\csv\\20220627_hdc_split_Fly1\\et"

savepath = os.path.join(datadir,"all_info_eb_fb.csv")


data = pd.read_csv(savepath)


#%% repair csv data  as is a nightmare
w_eb_o = data['wedges_eb']
w_fb_u_o = data['wedges_fb_upper']
w_fb_l_o = data['wedges_fb_lower']

# Repair string data
for i,e in enumerate(w_eb_o):
    enum_str = e.replace('[','').replace(']','').replace('\n','').split()
    enum = [float(num) for num in enum_str]
    if i==0:
        w_eb = np.empty([len(w_eb_o), len(enum)])
    w_eb[i,:] =enum

for i,e in enumerate(w_fb_u_o):
    enum_str = e.replace('[','').replace(']','').replace('\n','').split()
    enum = [float(num) for num in enum_str]
    if i==0:
        w_fb_u = np.empty([len(w_fb_u_o), len(enum)])
    w_fb_u[i,:] =enum    

for i,e in enumerate(w_fb_l_o):
    enum_str = e.replace('[','').replace(']','').replace('\n','').split()
    enum = [float(num) for num in enum_str]
    if i==0:
        w_fb_l = np.empty([len(w_fb_l_o), len(enum)])
    w_fb_l[i,:] =enum 
# Check looks ok    
plt.figure()
plt.imshow(w_eb,aspect='auto',interpolation='none')
plt.show()

plt.figure()
plt.imshow(w_fb_u,aspect='auto',interpolation='none')
plt.show()

plt.figure()
plt.imshow(w_fb_l,aspect='auto',interpolation='none')
plt.show()
# %% Product of EPG against hDelta
# looks for congruence and offset of signals
# Will give a 1D read out of coherence to overlay with behavioural data
ns = np.shape(w_eb)
e_2_Fu = np.empty([ns[1], ns[0]])
e_2_Fl = np.empty([ns[1], ns[0]])
xi = np.linspace(0,ns[1]-1,ns[1],dtype='int')
for i in range(ns[1]):
    X = w_eb[:,xi]
    X = np.subtract(X,np.mean(X,axis =1).reshape(-1,1))
    
    Y = w_fb_u
    Y = Y-np.mean(Y,axis=1).reshape(-1,1)
    x2 = np.matmul(X,np.transpose(Y))
    e_2_Fu[i,:] = np.diag(x2)
    
    Y = w_fb_l
    Y = Y-np.mean(Y,axis=1).reshape(-1,1)
    x2 = np.matmul(X,np.transpose(Y))
    e_2_Fl[i,:] = np.diag(x2)
    
    xi = np.append(xi[1:],xi[0])
    print(xi)

e_2_Fu = e_2_Fu/np.max(e_2_Fu,axis = 0)
plt.figure()
X = w_eb
X = np.subtract(X,np.mean(X,axis =1).reshape(-1,1))
plt.imshow(np.transpose(X),aspect='auto',interpolation='none')

plt.figure()
plt.imshow(e_2_Fu,aspect='auto',interpolation='none')
plt.show()

plt.figure()
plt.imshow(e_2_Fl,aspect='auto',interpolation='none')
plt.show()
#%% Plot behaviour
x = data['x']
y = data['y']
colour = data['fitted_amplitude_fb_lower']
colour = data['offset_phase_fb_upper']-data['offset_phase_eb']
ip = data['instrip'] 
inplume = []
for i,p in enumerate(ip):
    try:
        b = float(p)>0
    except:
        b = 0
    if b>0:
        inplume = np.append(inplume,i)
c_map = plt.get_cmap('coolwarm')

cnorm = mpl.colors.Normalize(vmin=-3, vmax=3)
scalarMap = cm.ScalarMappable(cnorm, c_map)
c_map_rgb = scalarMap.to_rgba(colour)
x = x-x[0]
y = y -y[0]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x[inplume],y[inplume],color=[0.5, 0.5, 0.5])
for i in range(len(x)-1):
    ax.plot(x[i:i+2],y[i:i+2],color=c_map_rgb[i+1,:3])
#%%

#%%
plt.close('all')
t = data['seconds']
h = data['heading']
s = data['instrip']
eb = data['offset_phase_eb']
fb = data['offset_phase_fb_upper']
db = fn.wrap(fb-eb)
c = data['fitted_amplitude_fb_upper']
plt.figure()
plt.plot(t,h,color='k')
plt.plot(t,eb,color=[0.5,0.5,1])
#plt.plot(t,fb,color='r')
plt.scatter(t,fb,s=c*100,color='r')
plt.plot(t,s,color=[0.3,0.3,0.3])

plt.show()
plt.figure()

plt.plot(t,h,color='k')
plt.scatter(t,db,s=c*100)
plt.plot(t,s,color=[0.3,0.3,0.3])
plt.show()
#%% Plot phase and and amplitude at plume entry

rootdir = "Y:\\Data\\FCI\\AndyData\\hDeltaC_imaging\\csv"
folders = os.listdir(rootdir)
savedir = "Y:\Data\FCI\FCI_summaries\hDeltaC\To_plume_transition"
pi = np.pi
for fly,f in enumerate(folders):
    # Load data
    print(fly)
    loadpath = os.path.join(rootdir,f,"et\\all_info_eb_fb.csv")


    data = pd.read_csv(loadpath)

    s = pd.Series.to_numpy(data['instrip'])
    t = data['seconds']-data['seconds'][0]
    heading = data['heading']
    phase = data['offset_phase_fb_upper']
    phase_eb =data['offset_phase_eb']
    rel_phase = fn.wrap(phase-phase_eb)
    phase_eb = pd.Series.to_numpy(phase_eb)
    rel_phase = pd.Series.to_numpy(rel_phase)
    power = data['fitted_amplitude_fb_upper']
    s[s=='False'] = 0
    s2 = np.array([float(si) for si in s])
    sd = np.diff(s2)
    ondx = [i+1 for i,si in enumerate(sd) if si>0]
    entry_an = np.empty(len(ondx))
    on_point = 7
    off_point = 5
    num_incl = np.sum(t<on_point)
    num_extr = np.sum(t<(on_point+off_point))-num_incl
    t_plot = t[t<(on_point+off_point)]
    mn_head = np.empty(len(ondx))
    entry_phase = np.empty([len(ondx), num_incl+num_extr])
    entry_power = np.empty([len(ondx), num_incl+num_extr])
    entry_phase_corr = np.empty([len(ondx), num_incl+num_extr])
    entry_phase_corr_p = np.empty([len(ondx), num_incl+num_extr])
    entry_phase_r_epg = np.empty([len(ondx), num_incl+num_extr]) 
    entry_phase_epg = np.empty([len(ondx), num_incl+num_extr]) 
    for i,dx in enumerate(ondx):
        ent_array = np.linspace(dx-3,dx+3,7,dtype=int)
        p_array = np.linspace(dx-(num_incl-1),dx+num_extr,num_incl+num_extr,dtype=int)
               
        if np.max(p_array)>len(phase):
            p_array = p_array[p_array<len(phase)]
                     
        mn_head[i] = stats.circmean(heading[ent_array],high=pi,low=-pi,)
        entry_phase[i,:len(p_array)] = phase[p_array]
        entry_power[i,:len(p_array)] = power[p_array]
        entry_phase_corr[i,:len(p_array)] = fn.wrap(phase[p_array]-mn_head[i])
        entry_phase_r_epg[i,:len(p_array)] =  rel_phase[p_array]
        entry_phase_epg[i,:len(p_array)] =  phase_eb[p_array]
        if i>0:
            entry_phase_corr_p[i,:len(p_array)] =fn.wrap(phase[p_array]-mn_head[i-1])
       
            
            
    
    
    m_ep_corr = stats.circmean(entry_phase_corr[1:,:],high=pi,low=-pi,axis =0)
    m_ep   = stats.circmean(entry_phase[1:,:],high=pi,low=-pi,axis =0)
    m_ep_corr_p = stats.circmean(entry_phase_corr_p[1:,:],high=pi,low=-pi,axis=0)
    m_ep_rel = stats.circmean(entry_phase_r_epg[1:,:],high=pi,low=-pi,axis=0)
    m_ep_epg = stats.circmean(entry_phase_epg[1:,:],high=pi,low=-pi,axis=0)
    c_phase = np.cos(entry_phase)
    c_phasemn = np.mean(c_phase[1:,:],axis=0)
    
    
    yt = [-pi,-pi/2,0,pi/2,pi]
    
    
    plt.close('all')
    plt.figure()
    plt.plot(t_plot,np.transpose(c_phase[1:,:]),color=[0.5, 0.5, 0.5])
    plt.plot(t_plot,c_phasemn,color='k')
    plt.plot([min(t_plot), max(t_plot)],[0,0],color='k',linestyle='--')
    plt.plot([on_point,on_point],[-pi,pi],color='r')
    
    plt.ylim([-1,1])
    plt.show()
    plt.ylabel('Cos phase')
    plt.savefig(os.path.join(savedir,'Phase_cos' + str(fly) + '.png'))
    
    
    plt.figure()
    ax = plt.gca()
    plt.plot(t_plot,np.transpose(entry_phase_r_epg[1:,:]),color=[0.5, 0.5, 0.5])
    plt.plot(t_plot,m_ep_rel,color='k')
    plt.plot([min(t_plot), max(t_plot)],[0,0],color='k',linestyle='--')
    plt.plot([on_point,on_point],[-pi,pi],color='r')
    plt.yticks(yt,labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
    ax.yaxis.set_major_formatter('$\pi$')
    plt.ylabel('Phase relative to EPG')
    plt.ylim([-pi,pi])
    plt.show()
    plt.savefig(os.path.join(savedir,'Phase_rel_epg' + str(fly) + '.png'))
    
   
    plt.figure()
    plt.plot(t_plot,np.transpose(entry_phase_corr[1:,:]),color=[0.5, 0.5, 0.5])
    plt.plot(t_plot,m_ep_corr,color='k')
    plt.plot([min(t_plot), max(t_plot)],[0,0],color='k',linestyle='--')
    plt.plot([on_point,on_point],[-pi,pi],color='r')
    plt.yticks(yt,labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
    ax.yaxis.set_major_formatter('$\pi$')
    plt.ylabel('Phase relative to entry heading')
    plt.ylim([-pi,pi])
    plt.show()
    plt.savefig(os.path.join(savedir,'Phase_rel_entry' + str(fly) + '.png'))
    
    plt.figure()
    plt.plot(t_plot,np.transpose(entry_phase_corr_p[1:,:]),color=[0.5, 0.5, 0.5])
    plt.plot(t_plot,m_ep_corr_p,color='k')
    plt.plot([min(t_plot), max(t_plot)],[0,0],color='k',linestyle='--')
    plt.plot([on_point,on_point],[-pi,pi],color='r')
    plt.yticks(yt,labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
    ax.yaxis.set_major_formatter('$\pi$')
    plt.ylabel('Phase relative to prior entry heading')
    plt.ylim([-pi,pi])
    plt.show()
    plt.savefig(os.path.join(savedir,'Phase_rel_prior_entry' + str(fly) + '.png'))
    
    plt.figure()
    plt.plot(t_plot,m_ep_rel,color='k')
    plt.plot(t_plot,m_ep,color=[0.5, 0.5, 1])
    plt.plot(t_plot,m_ep_epg,color=[1, 0.5, 0.5])
    plt.plot([min(t_plot), max(t_plot)],[0,0],color='k',linestyle='--')
    plt.plot([on_point,on_point],[-pi,pi],color='r')
    plt.yticks(yt,labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
    ax.yaxis.set_major_formatter('$\pi$')
    plt.ylabel('Phase relative to EPG')
    plt.ylim([-pi,pi])
    plt.show()
    plt.savefig(os.path.join(savedir,'Phase_rel_epg_mean' + str(fly) + '.png'))
    
    plt.figure()
    plt.plot(t_plot,np.transpose(entry_phase[1:,:]),color=[0.5,0.5,0.5])
    plt.plot(t_plot,m_ep,color='k')
    plt.plot([min(t_plot), max(t_plot)],[0,0],color='k',linestyle='--')
    plt.plot([on_point,on_point],[-pi,pi],color='r')
    plt.yticks(yt,labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
    ax.yaxis.set_major_formatter('$\pi$')
    plt.ylabel('Phase')
    plt.ylim([-pi,pi])
    plt.show()
    plt.savefig(os.path.join(savedir,'Phase_' + str(fly) + '.png'))
    
    
    
    plt.figure()
    plt.plot(t_plot,np.transpose(entry_power[1:,:]),color=[0.5,0.5,0.5])    
    plt.plot(t_plot,np.mean(np.transpose(entry_power[1:,:]),axis=1),color='k')    
    plt.plot([on_point,on_point],[0,np.max(entry_power[:])],color='r')
    plt.ylabel('Bump Amp')
    plt.savefig(os.path.join(savedir,'Bump_Amp' + str(fly) + '.png'))
    plt.show()
    
    
    
   
#%% scatters of mean phase for entry versus exit angles for t-1, t0, t+1
rootdir = "Y:\\Data\\FCI\\AndyData\\hDeltaC_imaging\\csv"
folders = os.listdir(rootdir)
savedir = "Y:\Data\FCI\FCI_summaries\hDeltaC\To_plume_transition"
plt.close('all')
pi = np.pi
yt = [-pi,-pi/2,0,pi/2,pi]
for fly,f in enumerate(folders):
    # Load data
    print(fly)
    loadpath = os.path.join(rootdir,f,"et\\all_info_eb_fb.csv")
    data = pd.read_csv(loadpath)

    s = pd.Series.to_numpy(data['instrip'])
    t = data['seconds']-data['seconds'][0]
    heading = data['heading']
    phase = data['offset_phase_fb_upper']
    phase_eb =data['offset_phase_eb']
    rel_phase = fn.wrap(phase-phase_eb)
    phase_eb = pd.Series.to_numpy(phase_eb)
    rel_phase = pd.Series.to_numpy(rel_phase)
    power = data['fitted_amplitude_fb_upper']
    s[s=='False'] = 0
    s2 = np.array([float(si) for si in s])
    sd = np.diff(s2)
    ondx = [i+1 for i,si in enumerate(sd) if si>0]
    offdx = [i+1 for i,si in enumerate(sd) if si<0]
    entry_an = np.empty(len(ondx))
    on_point = float(0.5)
    num_incl = np.sum(t<on_point)
    entry_phase = np.empty(len(ondx))
    phase_amp = np.empty(len(ondx))
    entry_phase_epg = np.empty(len(ondx))
    entry_heading = np.empty(len(ondx))
    e_time = np.zeros(len(ondx))
    e_in = np.zeros(len(ondx))
    e_num = np.linspace(0,len(ondx)-1,len(ondx))
    for i,dx in enumerate(ondx):
        t_dx = dx-num_incl
        entry_phase[i] = stats.circmean(phase[t_dx:dx],high=pi,low=-pi)
        phase_amp[i] = np.mean(power[t_dx:dx])
        entry_phase_epg[i] = stats.circmean(phase_eb[t_dx:dx],high=pi,low=-pi)
        entry_heading[i] = stats.circmean(heading[t_dx:dx],high=pi,low=-pi)
        try:
            e_in[i] = t[offdx[i]]-t[dx]
        except:
            e_in
        if i>0:
            e_time[i] = t[dx]-t[ondx[i-1]]
    entry_phaseO = entry_phase.copy()
    phase_error = np.abs(fn.wrap(entry_heading-entry_phase))        
    entry_phase = entry_phase*np.sign(entry_heading)
    entry_phase_epg = entry_phase_epg*np.sign(entry_heading)
    e_time[e_time>30] = 30
    # plt.figure()
    # plt.plot([0, 0],[-pi,pi],color='k',linestyle='--')
    # plt.plot([-pi,pi],[-pi, pi],color='k',linestyle='--')
    # plt.scatter(entry_phase_epg,entry_phase,s=phase_amp*100,c=e_time)
    # plt.ylabel('hDelta_phase')
    # plt.xlabel('EPG phase')
    # plt.yticks(yt,labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
    # plt.xticks(yt,labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
    # plt.show()
    
    plt.figure()
    x = data['x']
    y = data['y']
    plt.plot(x,y,color='k')
    plt.scatter(x[ondx],y[ondx],c= entry_phaseO,vmin=-pi,vmax=+pi,cmap='coolwarm',s = phase_amp*1000)
    plt.gca().set_aspect('equal')
    plt.show()
    # plt.figure()
    # plt.plot([0, 0],[-pi,pi],color='k',linestyle='--')
    # plt.plot([-pi,pi],[-pi, pi],color='k',linestyle='--')
    # plt.scatter(entry_heading,entry_phase,s=phase_amp*100,c=e_time)
    # plt.ylabel('hDelta_phase')
    # plt.xlabel('Entry Heading')
    # plt.yticks(yt,labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
    # plt.xticks(yt,labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
    # plt.show()
#%%
datadir = "Y:\\Data\\FCI\\AndyData\\hDeltaC_imaging\\pickles\\20220627_hdc_split_Fly1\\et"
#"Y:\Data\FCI\AndyData\hDeltaC_imaging\pickles\20220627_hdc_split_Fly1\et\get_all_info_eb_fb.p"
savepath = os.path.join(datadir,"get_all_info_eb_fb.p")
with open(savepath,'rb') as f:
    data = pkl.load(f)