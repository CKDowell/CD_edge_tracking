# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:25:38 2026

@author: dowel
"""

import numpy as np
import pandas as pd
import analysis_funs.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
import pickle 
from Utilities.utils_general import utils_general as ug

meta_data = {'stim_type': 'alternation',
              'act_inhib':'inhib',
              'ledOny': 600,
              'ledOffy':'all',
              'ledOnx': -1000,
              'ledOffx': 1000,
              'LEDoutplume': True,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 0,
            
              }

#%%
rootdir = r'E:\Data\Optogenetics\KatieData\hDeltaA\GtACR1_HorizontalPlumeLEDAlt'
plt.close('all')
flies = [
    r'260612\f2\Trial2',
    r'260612\f2\Trial3',
    r'260612\f3\Trial3',
    r'260616\f1\trial1',
    r'260616\f1\trial2',
   # r'260616\f2\trial1',# Stim not good ask Katie
   # r'260616\f2\trial2',# Stim not good ask Katie
   # r'260616\f2\trial3',# Stim not good ask Katie
    r'260616\f2\trial4',
    r'260616\f2\trial5',
    r'260616\f3\trial1',
    r'260616\f3\trial2',
   # r'260616\f4\trial1',# Fly is not walking straight
    #r'260616\f4\trial2',# Fly is not walking straight
    r'260616\f5\trial1', # Fly not walking that straight but still looks like an effect, consider discarding?
    ]

# Notes for Katie:
    # 1 She needs to run 11 trials
    # 2 She needs to keep running the ET assay



meta_data['stim_type'] = 'pulse'
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure()
    op.plot_plume_simple(meta_data,df)
    plt.title(f)
    
#%% Plot stim segments
plt.close('all')
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    led = df['led1_stpt'].to_numpy()
    ledon = led==0
    blks,bsz = ug.find_blocks(ledon)
    plt.figure()
    t = ug.get_ft_time(df)
    dt = np.mean(np.diff(t))
    ts = int(20/dt)
    x = df['ft_posx'].to_numpy()
    y = df['ft_posy'].to_numpy()
    x,y = ug.fictrac_repair(x, y)
    yoff=0
    xoff = 0
    for i2, b in enumerate(blks):
        dx = np.arange(b-ts,b+bsz[i2]+ts)
        dx = dx[dx<len(x)]
        tx =x[dx]-x[b-ts]
        ty = y[dx]-y[b-ts]
        yoff= yoff-np.max(ty)
        tled = ledon[dx]
        plt.plot(tx+xoff,ty+yoff,color='k')
        plt.scatter(tx[tled]+xoff,ty[tled]+yoff,color=[0.6,1,.6])
        yoff+= -50   +np.min(ty)
        #xoff += 50
    g= plt.gca()
    g.set_aspect('equal')
    plt.title(f)
    
#%%

rootdir = r'E:\Data\Optogenetics\KatieData\hDeltaJ\GtACR1_HorizontalPlumeLEDAlt'
plt.close('all')
flies = [
    r'260610\f1\Trial2',
    #r'260610\f5\Trial1',
    #r'260610\f5\Trial2'
    r'260618\f1\Trial2',
    r'260618\f2\Trial1',
    r'260618\f3\Trial1',
    r'260618\f3\Trial3',
    r'260618\f4\Trial1'
    ]
meta_data['stim_type'] = 'alternation_jump'
savedir = 'D:\RoughPlots\hDeltaJ'
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure()
    op.plot_plume_simple(meta_data,df)
    plt.title(f)
    plt.savefig(os.path.join(savedir,'EgTraj_'+str(i)+'.png'))
#%%
rootdir = r'E:\Data\Optogenetics\KatieData\hDeltaJ\GtACR1_HorizontalPlumeLEDAlt'
plt.close('all')
flies = [
    r'260618\f1\Trial3',
    r'260618\f2\Trial2',
    r'260618\f3\Trial2',
    r'260618\f4\Trial2'
    ]
meta_data['stim_type'] = 'pulse'
savedir = 'D:\RoughPlots\hDeltaJ'
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure()
    op.plot_plume_simple(meta_data,df)
    plt.title(f)
   