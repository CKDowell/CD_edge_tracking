# -*- coding: utf-8 -*-
"""
Created on Thu May  8 09:46:19 2025

@author: dowel


"""

#%%
import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
from Utilities.utils_general import utils_general as ug
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import locally_linear_embedding as LLE
import dtw
from scipy import stats
#%%
rootdir = r'Y:\Data\Behaviour\AndyVanilla\new_45_and_90_degree_plume_data'

searchdir = os.path.join(rootdir)
indir = os.listdir(searchdir)
plt.close('all')
for i,f in enumerate(indir):
    loaddir = os.path.join(rootdir,f)
    ft = fc.read_log(loaddir)
    x = ft['ft_posx'].to_numpy()
    y = ft['ft_posy'].to_numpy()
    x,y = ug.fictrac_repair(x,y)
    ins = ft['instrip'].to_numpy()
    pstart = np.where(ins)[0][0]
    x = x-x[pstart]
    y = y-y[pstart]
    xblack = x.copy()
    yblack = y.copy()
    xred = x.copy()
    yred = y.copy()
    
    
    
    xblack[ins] = np.nan
    yblack[ins] = np.nan
    yred[~ins] = np.nan
    xred[~ins] = np.nan
    plt.figure()
    if 'horizontal' in f:
        minx = np.min(x[pstart:])
        maxx = np.max(x[pstart:])
        plt.fill_between([minx,maxx],[0,0],[50,50],color=[0.7,0.7,0.7])
    elif '45' in f:
        xstart = -25*np.cos(np.pi/4)
        xstart2 = 25*np.cos(np.pi/4)
        ystart = 25*np.sin(np.pi/4)
        ystart2 = -25*np.sin(np.pi/4)
        yend = np.max(x) + 25*np.sin(np.pi/4)
        yend2 = np.max(x) - 25*np.sin(np.pi/4)
        xend = np.max(x)-25*np.cos(np.pi/4)
        xend2 = np.max(x)+25*np.cos(np.pi/4)
        plt.fill([xstart,xend,xend2,xstart2,xstart],[ystart,yend,yend2,ystart2,ystart],color=[0.7,0.7,0.7]) 
    plt.plot(xblack,yblack,color='k')
    plt.plot(xred,yred,color='r')
    
    plt.gca().set_aspect('equal')
    plt.title(f)
    plt.show()
    
    
    
    loaddir = os.path.join(rootdir,f)
    ft = fc.read_log(loaddir)
    ins = ft['instrip']
    heading = ft['ft_heading']
    tt = ug.get_ft_time(ft)
    tdiff = np.mean(np.diff(tt))
    xnum = np.round(0.25/tdiff).astype('int')
    
    s_start,s_size = ug.find_blocks(ins,merg_threshold = xnum)
    s_start = s_start[s_size>=xnum]
    s_size = s_size[s_size>=xnum]
    
    entry_angles = np.zeros(len(s_start))
    exit_angles = np.zeros(len(s_start))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for ib,s in enumerate(s_start):
        thead = heading[s-xnum:s]
        entry_angles[ib] = stats.circmean(thead,high = np.pi,low=-np.pi)
        se = s+s_size[ib]
        exit_angles[ib]  =stats.circmean(heading[se:se+xnum],high = np.pi,low=-np.pi)
        ax.plot([0,entry_angles[ib]],[0, 1],color='k',alpha=0.2)
        ax.plot([0,exit_angles[ib]], [0,1],color='r',alpha=0.2)
    
    emean = stats.circmean(entry_angles,high=np.pi,low=-np.pi)
    ax.plot([0,emean],[0, 1],color='k',linewidth=3)
    
    
    exmean = stats.circmean(exit_angles,high=np.pi,low=-np.pi)
    ax.plot([0,exmean], [0,1],color='r',linewidth=3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction('clockwise')
    plt.title(f)

    
#%% Example plumes entry vs exit angles

#%%
rootdir = r'Y:\Data\Behaviour\AndyVanilla\new_45_and_90_degree_plume_data'

searchdir = os.path.join(rootdir)
indir = os.listdir(searchdir)
plt.close('all')
for i,f in enumerate(indir):
    loaddir = os.path.join(rootdir,f)
    ft = fc.read_log(loaddir)
    ins = ft['instrip']
    heading = ft['ft_heading']
    tt = ug.get_ft_time(ft)
    tdiff = np.mean(np.diff(tt))
    xnum = np.round(0.25/tdiff).astype('int')
    
    s_start,s_size = ug.find_blocks(ins,merg_threshold = xnum)
    s_start = s_start[s_size>=xnum]
    s_size = s_size[s_size>=xnum]
    
    entry_angles = np.zeros(len(s_start))
    exit_angles = np.zeros(len(s_start))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for ib,s in enumerate(s_start):
        thead = heading[s-xnum:s]
        entry_angles[ib] = stats.circmean(thead,high = np.pi,low=-np.pi)
        se = s+s_size[ib]
        exit_angles[ib]  =stats.circmean(heading[se:se+xnum],high = np.pi,low=-np.pi)
        ax.plot([0,entry_angles[ib]],[0, 1],color='k',alpha=0.2)
        ax.plot([0,exit_angles[ib]], [0,1],color='r',alpha=0.2)
    
    emean = stats.circmean(entry_angles,high=np.pi,low=-np.pi)
    ax.plot([0,emean],[0, 1],color='k',linewidth=3)
    
    
    exmean = stats.circmean(exit_angles,high=np.pi,low=-np.pi)
    ax.plot([0,exmean], [0,1],color='r',linewidth=3)
    
    plt.title(f)
#%% example vertical plume
datadir = "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS60291\\240304\\f2\Trial1"
lname = os.listdir(datadir)
savepath = os.path.join(datadir,lname[0])
ft = fc.read_log(savepath)
ins = ft['instrip']
heading = ft['ft_heading']
tt = ug.get_ft_time(ft)
tdiff = np.mean(np.diff(tt))
xnum = np.round(0.25/tdiff).astype('int')

s_start,s_size = ug.find_blocks(ins,merg_threshold = xnum)
s_start = s_start[s_size>=xnum]
s_size = s_size[s_size>=xnum]

entry_angles = np.zeros(len(s_start))
exit_angles = np.zeros(len(s_start))
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

for ib,s in enumerate(s_start):
    thead = heading[s-xnum:s]
    entry_angles[ib] = stats.circmean(thead,high = np.pi,low=-np.pi)
    se = s+s_size[ib]
    exit_angles[ib]  =stats.circmean(heading[se:se+xnum],high = np.pi,low=-np.pi)
    ax.plot([0,entry_angles[ib]],[0, 1],color='k',alpha=0.2)
    ax.plot([0,exit_angles[ib]], [0,1],color='r',alpha=0.2)

emean = stats.circmean(entry_angles,high=np.pi,low=-np.pi)
ax.plot([0,emean],[0, 1],color='k',linewidth=3)


exmean = stats.circmean(exit_angles,high=np.pi,low=-np.pi)
ax.plot([0,exmean], [0,1],color='r',linewidth=3)
ax.set_theta_zero_location("N")
ax.set_theta_direction('clockwise')
