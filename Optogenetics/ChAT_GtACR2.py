# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 11:12:52 2026

@author: dowel
"""

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 
from pathlib import Path
from Utilities.utils_general import utils_general as ug
from scipy import stats
from scipy import signal as sg
#%%
rootdir = r'Y:\Data\Optogenetics\ChatGtACR2'

flies = [
     r'M3\260701\f1\Trial2',
     r'M4\260701\f2\Trial1',
     r'M5\260701\f3\Trial1'
        ]
plt.close('all')
savedir = rootdir
for i,f in enumerate(flies):
    sdir = os.path.join(rootdir, flies[i])
    lname = np.array( os.listdir(sdir))
    
    lname = ug.get_folder(sdir)   
    print(len(lname))
    datadir = os.path.join(sdir,lname[-1])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    t = ug.get_ft_time(df)
    u = ug()
    x  = df['ft_posx'].to_numpy(); y = df['ft_posy'].to_numpy()
    
    x = x-x[0];y = y-y[0]
    x = sg.savgol_filter(x, 21, 3)
    y = sg.savgol_filter(y, 21, 3)
    dx = np.diff(x);dy = np.diff(y)
    dd = np.sqrt(dx**2+dy**2)
    d =np.append(0, np.cumsum(dd))
    x,y = ug.fictrac_repair(x, y)
    dx,dy,dd = u.get_velocity( x, y, t)
    ledon = df['led1_stpt'].to_numpy()==0
    
    bstart,blen = ug.find_blocks(ledon)
    dt = int(1/np.mean(np.diff(t)))
    t1 = bstart[0]-dt*20
    t2 = bstart[4]+blen[4]+dt*20
    
    plt.figure()
    plt.plot(t[t1:t2]-t[t1],dd[t1:t2],color='k')
    #plt.plot(t,ledon*10,color='b')
    plt.fill_between(t[t1:t2]-t[t1],ledon[t1:t2]*0+15,ledon[t1:t2]*5+15,color=[.4,.4,1],linewidth=0)
    plt.ylim([-1,35])
    plt.xlim([0,t[t2]-t[t1]])
    plt.savefig(os.path.join(savedir,'Chat_'+str(i)+'.pdf'))
    
    
    plt.figure()
    plt.plot(x,y)
    plt.gca().set_aspect('equal')
    plt.figure()
    plt.plot(t,d,color='k')
    plt.scatter(t[ledon],d[ledon],color='b')