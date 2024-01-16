# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:11:26 2023

@author: dowel
"""

#%%
import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
#%% MBON21_stimulation
meta_data = {'stim_type': 'plume',
    'ledONy': 300,
             'ledOffy':600,
             'LEDoutplume': True,
             'LEDinplume': False,
             'PlumeWidth': float(50),
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231214\\f1\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231214\\f1\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231214\\f2\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231214\\f2\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231214\\f2\Trial3"]
sdir = savedirs[0]

lname = os.listdir(sdir)
savepath = os.path.join(sdir,lname[0])
df = fc.read_log(savepath)
#%% 
#%% Plot 1st run through
plt.close('all')
for sdir in savedirs:
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    opto.plot_plume(meta_data,df)
#%% No odour stim outside
plt.close('all')
meta_data = {'stim_type': 'plume',
             'act_inhib':'act',
    'ledONy': 'all',
             'ledOffy':'all',
             'LEDoutplume': True,
             'LEDinplume': False,
             'PlumeWidth': float(50),
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_ACV_stim_border\\231214\\f2\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation_no_odour\\231214\\f2\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation_no_odour\\231215\\f2\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation_no_odour\\231215\\f2\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation_no_odour\\231215\\f3\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation_no_odour\\231215\\f3\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation_no_odour\\231215\\f4\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation_no_odour\\231215\\f4\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation_no_odour\\231215\\f5\Trial1_30mm",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation_no_odour\\231215\\f5\Trial2_10mm",
            ]
plumewidths = [50,50,50,50,50,50,50,50,30,10]
pltsavedir = "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation_no_odour\SummaryFigures"
for i in range(len(savedirs)):
    sdir = savedirs[i]

    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    meta_data['PlumeWidth'] = plumewidths[i]
    op = opto()
    op.plot_plume_simple(meta_data,df)
    snames = sdir.split('\\')
    plt.title(snames[-3] + ' ' + snames[-2] + ' ' +snames[-1])
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.png'
    plt.savefig(os.path.join(pltsavedir,savename))

#%% ACV and stim outside without shifting plume location
plt.close('all')
meta_data = {'stim_type': 'plume',
             'act_inhib':'act',
    'ledONy': 300,
             'ledOffy':600,
             'LEDoutplume': True,
             'LEDinplume': False,
             'PlumeWidth': float(50),
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231215\\f2\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231215\\f3\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231215\\f4\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231215\\f5\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231219\\f1\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231219\\f1\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231219\\f2\Trial1"
            ]
pltsavedir = "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\SummaryFigures"
for i in range(len(savedirs)):
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    snames = sdir.split('\\')
    plt.title(snames[-3] + ' ' + snames[-2] + ' ' +snames[-1])
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.png'
    plt.savefig(os.path.join(pltsavedir,savename))
    plt.ylim([0,1000])
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '_1000.png'
    plt.savefig(os.path.join(pltsavedir,savename))
    plt.ylim([0,2000])
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '_2000.png'
    plt.savefig(os.path.join(pltsavedir,savename))
# %% Scatter metrics
plt.close('all')
titles = ['Max distance from plume','Time outside plume','Path length outside plume','Median velocity outside plume']
ylabs = ['Distance (mm)','time (s)', 'Mean path length (mm)', 'Velocity (mm/s)']

savedirs = ["Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231215\\f2\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231215\\f3\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231215\\f4\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231215\\f5\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231219\\f1\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\\231219\\f2\Trial1"
            ]
pltsavedir = "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_stimulation\SummaryFigures"
for i in range(len(savedirs)):
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op = opto()
    exp_dat = op.extract_stats(meta_data,df)
    t_dat = exp_dat['mean data']
    ts = np.shape(t_dat)
    
    for i2 in range(ts[1]-1):
        plt.figure(num=i2,figsize=(3,3))
        x = np.array([1.0, 2.0])
        y = t_dat[0:2,i2+1]
        plt.plot(x,y,color='k')
    print(y)
for i2 in range(ts[1]-1):
    plt.figure(num=i2,figsize=(3,3))
    plt.title(titles[i2])
    plt.xticks([1,2],labels=['No LED', 'LED'])
    plt.ylabel(ylabs[i2])
    if i2==1:
        plt.ylim([0,40])
    plt.subplots_adjust(left=0.2)
    savename = titles[i2] + '.png'
    plt.savefig(os.path.join(pltsavedir,savename))
# %% MBON21 inhibition
plt.close('all')
meta_data = {'stim_type': 'plume',
             'act_inhib':'inhib',
    'ledONy': 300,
             'ledOffy':600,
             'LEDoutplume': True,
             'LEDinplume': False,
             'PlumeWidth': float(50),
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_inhibition\\231221\\f1\\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_inhibition\\231221\\f1\\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_inhibition\\231222\\f1\\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_inhibition\\231222\\f2\\Trial1",
            ]
pltsavedir = "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_inhibition\SummaryFigures"
for i in range(len(savedirs)):
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op= opto()
    op.plot_plume_simple(meta_data,df)
    snames = sdir.split('\\')
    plt.title(snames[-3] + ' ' + snames[-2] + ' ' +snames[-1])
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.png'
    plt.savefig(os.path.join(pltsavedir,savename))
    plt.ylim([0,1000])
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '_1000.png'
    plt.savefig(os.path.join(pltsavedir,savename))
    plt.ylim([0,2000])
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '_2000.png'
    plt.savefig(os.path.join(pltsavedir,savename))
# %%
plt.close('all')
titles = ['Max distance from plume','Time outside plume','Path length outside plume','Median velocity outside plume']
ylabs = ['Distance (mm)','time (s)', 'Mean path length (mm)', 'Velocity (mm/s)']
savedirs = [
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_inhibition\\231221\\f1\\Trial2",#Better edge tracking in trial 2
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_inhibition\\231222\\f1\\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_inhibition\\231222\\f2\\Trial1",
            ]
pltsavedir = "Y:\Data\Optogenetics\MBONs\MBON21\\MBON21_inhibition\SummaryFigures"
for i in range(len(savedirs)):
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op = opto()
    exp_dat = op.extract_stats(meta_data,df)
    t_dat = exp_dat['mean data']
    ts = np.shape(t_dat)
    
    for i2 in range(ts[1]-1):
        plt.figure(num=i2,figsize=(3,3))
        x = np.array([1.0, 2.0])
        y = t_dat[0:2,i2+1]
        plt.plot(x,y,color='k')
    print(y)
for i2 in range(ts[1]-1):
    plt.figure(num=i2,figsize=(3,3))
    plt.title(titles[i2])
    plt.xticks([1,2],labels=['No LED', 'LED'])
    plt.ylabel(ylabs[i2])
    plt.subplots_adjust(left=0.2)
    savename = titles[i2] + '.png'
    plt.savefig(os.path.join(pltsavedir,savename))
#%% MBON 21 stimulaton pulses
plt.close('all')
meta_data = {'stim_type': 'pulse',
    'ledONy': 0,
             'ledOffy':0,
             'LEDoutplume': False,
             'LEDinplume': False,
             'PlumeWidth': False,
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON21_light_pulses\\231219\\f1\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_light_pulses\\231219\\f2\Trial1"]
for i in range(len(savedirs)):
    print(i)
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    #light_pulse_pre_post(meta_data,df)
#%% MBON 33 stimulation pulses
plt.close('all')
meta_data = {'stim_type': 'pulse',
    'ledONy': 0,
             'ledOffy':0,
             'LEDoutplume': False,
             'LEDinplume': False,
             'PlumeWidth': False,
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231219\\f4\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231219\\f4\Trial2_15s",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231219\\f5\Trial1_15s",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231220\\f1\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231220\\f1\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231220\\f2\Trial1"]
pltsavedir = "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\SummaryFigures"
for i in range(len(savedirs)):
    print(i)
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op = opto()
    op.light_pulse_pre_post(meta_data,df)
    snames = sdir.split('\\')
    plt.title(snames[-3] + ' ' + snames[-2] + ' ' +snames[-1])
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.png'
    plt.savefig(os.path.join(pltsavedir,savename))
    opto.plot_plume_simple(meta_data,df)
    savename = 'Traj_' + snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.png'
    plt.savefig(os.path.join(pltsavedir,savename))
#%% MBON 33 stimulation after edge trackingplt.close('all')
plt.close('all')
meta_data = {'stim_type': 'pulse',
    'ledONy': 0,
             'ledOffy':0,
             'LEDoutplume': False,
             'LEDinplume': False,
             'PlumeWidth': False,
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231220\\f1\Trial3_post_plume",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231220\\f2\Trial2_post_plume"
            ]
for i in range(len(savedirs)):
    print(i)
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    opto.plot_plume_simple(meta_data,df)
    #light_pulse_pre_post(meta_data,df)