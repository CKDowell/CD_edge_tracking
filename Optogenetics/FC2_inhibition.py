# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:15:45 2024

@author: dowel

These scripts analyse data from inhibition of FC2 neurons and the corresponding
control experiments

"""

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
import pickle 
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
#%% Visualise data - test animals
rootdir = 'Y:\\Data\\Optogenetics\\FC2\\FC2_maimon2_alternation_inhibition\\Test_Flies'
plt.close('all')
flies = [
    '240625\\f4\\Trial1',
         '240627\\f2\\Trial3',
         '240627\\f4\\Trial2',
         '240722\\f1\\Trial1',
         '240722\\f2\\Trial1',
         '240723\\f4\\Trial1',
         '240724\\f1\\Trial2',
         '240730\\f1\\Trial1']

for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
#%% Visualise control animals
rootdir = 'Y:\\Data\\Optogenetics\\FC2\\FC2_maimon2_alternation_inhibition\\Control_UAS_GtACR1'
flies = ['240722\\f3\\Trial1',
         '240724\\f2\\Trial2',
         '240730\\f2\\Trial1']
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
#%% Extract stats alternation experiment - Test flies
plt.close('all')
rootdir = 'Y:\\Data\\Optogenetics\\FC2\\FC2_maimon2_alternation_inhibition\\Test_Flies'
flies = [
    '240625\\f4\\Trial1',
         '240627\\f2\\Trial3',
         #'240627\\f4\\Trial2',# this animal did not return to the plume
         '240722\\f1\\Trial1',
         '240722\\f2\\Trial1',
         '240723\\f4\\Trial1',
         '240724\\f1\\Trial2',
         '240730\\f1\\Trial1'
         ]
xl = np.zeros(4)
colours = plt.cm.hsv(np.linspace(0, 1, len(flies)))
savedir = "Y:\\Data\\Optogenetics\\FC2\\FC2_maimon2_alternation_inhibition\\ProcessedData"
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    out_dict = op.extract_stats_alternation(meta_data,df)
    savename = os.path.join(datadir,'OptoFeatures.pkl')
    with open(savename,'wb') as fp:
        pickle.dump(out_dict,fp)
    sn = f.split('\\')
    sname = 'Test_'+ sn[0] +sn[1]+sn[2]+ '.pkl'
    savename = os.path.join(savedir,sname)
    with open(savename,'wb') as fp:
        pickle.dump(out_dict,fp)
#%% Extract stats alternation experiment - Control flies
plt.close('all')
rootdir = 'Y:\\Data\\Optogenetics\\FC2\\FC2_maimon2_alternation_inhibition\\Control_UAS_GtACR1'
flies = ['240722\\f3\\Trial1',
         '240724\\f2\\Trial2',
         '240730\\f2\\Trial1']
xl = np.zeros(4)
colours = plt.cm.hsv(np.linspace(0, 1, len(flies)))
savedir = "Y:\\Data\\Optogenetics\\FC2\\FC2_maimon2_alternation_inhibition\\ProcessedData"
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    out_dict = op.extract_stats_alternation(meta_data,df)
    savename = os.path.join(datadir,'OptoFeatures.pkl')
    with open(savename,'wb') as fp:
        pickle.dump(out_dict,fp)
    sn = f.split('\\')
    sname = 'Control_wtGtACR1_'+ sn[0] +sn[1]+sn[2]+ '.pkl'
    savename = os.path.join(savedir,sname)
    with open(savename,'wb') as fp:
        pickle.dump(out_dict,fp)
#%% Plot data
savedir = 'Y:\\Data\\Optogenetics\\FC2\\FC2_maimon2_alternation_inhibition\\ProcessedData'
indir = os.listdir(savedir)
plt.close('all')
control1_dx = [i for i,n in enumerate(indir) if 'Control_wtGtACR1' in n]
test_dx = [i for i,n in enumerate(indir) if 'Test' in n]
for i in test_dx:
    dataname = os.path.join(savedir,indir[i])
    
    out_dict = pd.read_pickle(dataname)
    x = out_dict['Data_ledON']
    y = out_dict['Data_ledOFF']
    dx = out_dict['Ratio_dx']
    rats = out_dict['Median Ratios']
    ratlog = out_dict['Median Ratios Log']
    x = x[dx,:]
    y = y[dx,:]
    for i2 in range(4):
        plt.figure(i2)
        xl[i2] = max(np.median(x[:,i2]),xl[i2])
        xl[i2] = max(np.median(y[:,i2]),xl[i2])
        plt.scatter(np.median(x[:,i2]),np.median(y[:,i2]),color='r')
        plt.xlim([0,xl[i2]])
        plt.ylim([0,xl[i2]])
        if i==len(flies)-1:
            plt.plot([0,xl[i2]],[0,xl[i2]],color='k',linestyle='--')
            plt.title(out_dict['Column Names'][i2])
            plt.xlabel('LED ON')
            plt.ylabel('LED Off')
        
        plt.figure(10)
        plt.scatter(i2,rats[i2],color='r')
        
for i in control1_dx:
    dataname = os.path.join(savedir,indir[i])
    
    out_dict = pd.read_pickle(dataname)
    x = out_dict['Data_ledON']
    y = out_dict['Data_ledOFF']
    dx = out_dict['Ratio_dx']
    rats = out_dict['Median Ratios']
    ratlog = out_dict['Median Ratios Log']
    x = x[dx,:]
    y = y[dx,:]
    for i2 in range(4):
        plt.figure(i2)
        xl[i2] = max(np.median(x[:,i2]),xl[i2])
        xl[i2] = max(np.median(y[:,i2]),xl[i2])
        plt.scatter(np.median(x[:,i2]),np.median(y[:,i2]),color='k')
        plt.xlim([0,xl[i2]])
        plt.ylim([0,xl[i2]])
        if i==len(flies)-1:
            plt.plot([0,xl[i2]],[0,xl[i2]],color='k',linestyle='--')
            plt.title(out_dict['Column Names'][i2])
            plt.xlabel('LED ON')
            plt.ylabel('LED Off')
        
        plt.figure(10)
        plt.scatter(i2+0.1,rats[i2],color='k')
        
plt.figure(10)
plt.yscale('log')
plt.ylabel('Ratios')
plt.xticks(np.arange(0,4),labels=out_dict['Column Names'])
plt.plot([0,3],[1,1],color='k',linestyle='--')
#%% Control flies
#     x = out_dict['Data_ledON']
#     y = out_dict['Data_ledOFF']
#     dx = out_dict['Ratio_dx']
#     rats = out_dict['Median Ratios']
#     ratlog = out_dict['Median Ratios Log']
#     x = x[dx,:]
#     y = y[dx,:]
#     for i2 in range(4):
#         plt.figure(i2)
#         xl[i2] = max(np.median(x[:,i2]),xl[i2])
#         xl[i2] = max(np.median(y[:,i2]),xl[i2])
#         plt.scatter(np.median(x[:,i2]),np.median(y[:,i2]),color=colours[i,:])
#         plt.xlim([0,xl[i2]])
#         plt.ylim([0,xl[i2]])
#         if i==len(flies)-1:
#             plt.plot([0,xl[i2]],[0,xl[i2]],color='k',linestyle='--')
#             plt.title(out_dict['Column Names'][i2])
#             plt.xlabel('LED ON')
#             plt.ylabel('LED Off')
        
#         plt.figure(10)
#         plt.scatter(i2,rats[i2],color=colours[i,:])
#         if i2 !=3:
#             plt.figure(100)
#             plt.scatter(i2,ratlog[i2],color=colours[i,:])
# plt.figure(10)
# plt.yscale('log')
# plt.plot([0,3],[1,1],color='k')
# plt.figure(100)
# plt.plot([0,3],[0,0],color='k')