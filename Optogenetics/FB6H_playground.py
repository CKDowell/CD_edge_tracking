# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:28:46 2025

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
plt.rcParams['pdf.fonttype'] = 42 
meta_data = {'stim_type': 'alternation',
              'act_inhib':'act',
              'ledOny': 700,
              'ledOffy':'all',
              'ledOnx': -1000,
              'ledOffx': 1000,
              'LEDoutplume': True,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 0,
            
              }
#%% Blanket inhibition - Test
plt.close('all')
rootdir = r'Y:\Data\Optogenetics\FB6H_SS95649\FB6H_inhibition'
flies = [r'Test\250213\f1\Trial1',# Good tracker, walked off
         #r'Test\250214\f1\Trial1', # Did not make inhib
         r'Test\250214\f3\Trial1',# Decent tracker, no obvious phen
         r'Test\250214\f5\Trial1',# Decent tracker, long returns
         r'Test\250218\f1\Trial1', # Strange tracker - not ET
         r'Test\250225\f1\Trial1', # Spiraling after stimulation, poor returns
         r'Test\250225\f3\Trial1', # Walked off
         r'Test\250227\f1\Trial1', # Walked off, plume cross overs
         #r'Test\250227\f3\Trial1', # Did not make inhib
         r'Test\250227\f5\Trial1', # Very long returns, probably not an ET
         r'Test\250228\f1\Trial1', # Good tracker then affected
         #r'Test\250228\f3\Trial1',
         #r'Test\250228\f5\Trial1',
         r'Test\250304\f1\Trial1', # ET, hard to tell if there is an effect
         r'Test\250304\f4\Trial1',# Walked off
         r'Test\250305\f1\Trial1', # Long returns
         r'Test\250306\f2\Trial1', # Normal ET
         r'Test\250306\f5\Trial1',# Walked off
         r'Test\250307\f2\Trial1',# Walked off
         
         ]
meta_data['act_inhib'] = 'inhib'
savedir = r'Y:\Data\Optogenetics\Summaries\FB6H'
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    plt.title(f)
    savename = f.split('\\')
    savename = "".join(savename)
    plt.savefig(os.path.join(savedir,savename + '.pdf'))
    plt.savefig(os.path.join(savedir,savename + '.png'))
#%% Blanket inhibition - Control
plt.close('all')
rootdir = r'Y:\Data\Optogenetics\FB6H_SS95649\FB6H_inhibition'
flies = [
    #r'Control\250213\f2\Trial1',
        # r'Control\250214\f2\Trial1',
         r'Control\250214\f4\Trial1', # Cross over not ET
         #r'Control\250218\f2\Trial1',
         r'Control\250225\f2\Trial1', #Good tracker
         #r'Control\250227\f2\Trial1',
         r'Control\250227\f4\Trial1',# Not ET
         r'Control\250228\f2\Trial1',#Not ET
         r'Control\250228\f4\Trial1',# ET
        # r'Control\250304\f2\Trial1',
        # r'Control\250304\f3\Trial1',
        # r'Control\250304\f5\Trial1',
        # r'Control\250304\f6\Trial1',
         r'Control\250305\f1\Trial1',# Experiment not run for long enough
         r'Control\250305\f2\Trial1',# ET, some change after light
        # r'Control\250306\f1\Trial1',
         r'Control\250306\f3\Trial1',# Experiment not run for long enough
         r'Control\250306\f4\Trial1',#No ET before light, ET after light
         #r'Control\250306\f6\Trial1',
        # r'Control\250306\f7\Trial1',
         #r'Control\250306\f8\Trial1',
         ]

meta_data['act_inhib'] = 'inhib'
for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    plt.title(f)

#%% Entry count alt
plt.close('all')
flies = [r'Test\250311\f4\Trial1',# Entry count alt
r'Test\250312\f2\Trial1',#Entry count alt inhib did not make it back
r'Test\250319\f1\Trial1', # Inhib did not make it back
r'Test\250319\f2\Trial1',# Inhib did not make it back


r'Test\250319\f3\Trial1', # Shorter returns with inhib
r'Test\250320\f1\Trial1', # Not enough entries
r'Test\250320\f2\Trial1',# Downwind tracker
r'Test\250320\f3\Trial1',# Longer plume returns on inhib
]
rootdir = r'Y:\Data\Optogenetics\FB6H_SS95649\FB6H_inhibition'
meta_data['stim_type'] = 'alternation'
meta_data['act_inhib'] = 'inhib'

for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    plt.title(f)
    


#%% Blanket excitation
savedir = r'Y:\Data\Optogenetics\Summaries\FB6H\Activation'
meta_data['act_inhib'] = 'act'
plt.close('all')
rootdir = r'Y:\Data\Optogenetics\FB6H_SS95649\FB6H_activation'
flies_test = [r'Test\250408\f1\Trial1', # not really ET before act
         #r'Test\250410\f1\Trial1', # Did not reach stim
         r'Test\250410\f3\Trial1', # ET then wandered off. +
         r'Test\250410\f5\Trial1', # Did not ET, but reached stim
        # r'Test\250423\f1\Trial1', # Did not reach stim
         r'Test\250423\f2\Trial1', # Some ET, reached stim outside plume, wandered off + (caveat, not walking much)
        # r'Test\250424\f1\Trial1', # Did not reach stim
         r'Test\250424\f3\Trial1', # In plume for pre stim, some ET with long returns after stim +-
         r'Test\250425\f1\Trial1', # No ET, not walking great, reached stim and did not ET
         r'Test\250425\f3\Trial1', # Reached stim, good edge tracking -
         r'Test\250426\f1\Trial1', # Not much walking outside plume, got to stim, seems like ET and unaffected -
         r'Test\250426\f3\Trial1', # ET before and after, walking in plume a lot -
        # r'Test\250429\f4\Trial1', # Did not reach stim zone
         r'Test\250429\f6\Trial1', # Did not leave plume before stim, but relatively normal ET after that -
        # r'Test\250430\f2\Trial1', # Did not reach stim zone
         r'Test\250430\f4\Trial1', # Long returns to plume, not really ET, unaffected by stim -
         r'Test\250430\f6\Trial1', # Good ET before and after no effect -
         r'Test\250501\f1\Trial1',#Tracked whole way but moved further from plume +-
         r'Test\250501\f3\Trial1',# Left after stim +
         r'Test\250503\f2\Trial1',
         r'Test\250503\f4\Trial1',
         r'Test\250506\f3\Trial1'
    ]

for i,f in enumerate(flies_test):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    plt.title(f + ' '+ str(i))
    savename = f.split('\\')
    savename = "".join(savename)
    plt.xlim([-400,400])
    plt.ylim([-200, 2500])
    plt.savefig(os.path.join(savedir,savename + '.pdf'))
    plt.savefig(os.path.join(savedir,savename + '.png'))
    
    plt.figure(101)
    op.opto_raster(meta_data,df,offset=i)
    
plt.plot([0,0],[0,i+1],color='k',linestyle='--')

#%% Blanket excitation - controls

meta_data['act_inhib'] = 'act'
plt.close('all')
rootdir = r'Y:\Data\Optogenetics\FB6H_SS95649\FB6H_activation'
flies_control = [r'Control_w\250410\f2\Trial1', # wondered off
         #r'Control_w\250410\f4\Trial1',
        # r'Control_w\250410\f6\Trial1', # Did not make stim
         r'Control_w\250423\f2\Trial1', # Strange tracker and wondered off
         r'Control_w\250423\f4\Trial1',# Strange tracker and wondered off
         r'Control_w\250426\f2\Trial1',# Excellent tracker
         r'Control_w\250426\f4\Trial1',# Excellent tracker
        # r'Control_w\250428\f2\Trial1',# No data
         r'Control_w\250429\f5\Trial1',# Good tracker
        # r'Control_w\250430\f3\Trial1',# Fictrac failed too much  
        r'Control_w\250501\f2\Trial1',# Good tracker
        #r'Control_w\250501\f4\Trial1', Fly barely walked
        #r'Control_w\250501\f5\Trial1', # Did not make to stim
        r'Control_w\250503\f1\Trial1',
        r'Control_w\250503\f3\Trial1',
       r'Control_w\250506\f1\Trial1',
        r'Control_w\250506\f2\Trial1',
       # r'Control_w\250530\f1\Trial1', Fly did not walk
        r'Control_w\250530\f2\Trial1', # Tracked length with long in odour passages
        r'Control_w\250530\f3\Trial1',# Tracked length with long in odour passages
        r'Control_w\250530\f4\Trial1',# Tracked length with long in odour passages
        r'Control_w\250530\f5\Trial1',# Tracked length with long in odour passages
    ]

for i,f in enumerate(flies_control):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    plt.title(f + ' '+ str(i))
    
    savename = f.split('\\')
    savename = "".join(savename)
    plt.xlim([-400,400])
    plt.ylim([-200, 2500])
    plt.savefig(os.path.join(savedir,savename + '.pdf'))
    plt.savefig(os.path.join(savedir,savename + '.png'))

    plt.figure(101)
    op.opto_raster(meta_data,df,offset=i)
    
plt.plot([0,0],[0,i+1],color='k',linestyle='--')



#%% Compare test and control stim
plt.close('all')
labmeetingdir = r'Y:\Presentations\2025\06_LabMeeting\FB6H'
yoff = 0
for i,f in enumerate(flies_test):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure(101)
    op.opto_raster(meta_data,df,offset=i)
    plt.figure(10+i)
    op.opto_return_raster(meta_data,df,offset=0)
    yoff = yoff+1000
    
plt.figure(101)
plt.plot([0,0],[0,i+1],color='k',linestyle='--')
i2 = i+2
for i,f in enumerate(flies_control):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure(101)
    op.opto_raster(meta_data,df,offset=i+i2)
    plt.figure(150+i)
    op.opto_return_raster(meta_data,df,offset=0)
    yoff = yoff+1000

    
plt.figure(101)
plt.plot([0,0],[i2,i2+i+1],color='k',linestyle='--')
plt.savefig(os.path.join(labmeetingdir,'SummaryTime.pdf'))
plt.savefig(os.path.join(labmeetingdir,'SummaryTime.png'))

#%% plot first stim
plt.close('all')
xoff = 0
plt.figure(figsize=(20,8))
for i,f in enumerate(flies_test):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    #plt.figure()
    op.plot_first_stim(df,xoffset=xoff)
    xoff = xoff+400
    
plt.xlim([-200,len(flies_test)*400])
plt.savefig(os.path.join(savedir,'FirstStim_test.pdf')) 

plt.figure(figsize=(20,8))
xoff = 0


for i,f in enumerate(flies_control):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    #plt.figure()
    op.plot_first_stim(df,xoffset=xoff)
    xoff = xoff+400
plt.xlim([-200,len(flies_test)*400])
plt.savefig(os.path.join(savedir,'FirstStim_control.pdf'))
#%% extract 
plt.close('all')
statchoice = ['mean dist','mean ret time','returns per m','run length','first return','return status']
test_data = np.empty((len(flies_test),len(statchoice)))
for i,f in enumerate(flies_test):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    xdat = op.extract_stats_threshold(df,700,statchoice)
    plt.scatter(np.arange(0,len(xdat))+np.random.rand(len(xdat))*0.05,xdat,color='r')
    test_data[i,:] = xdat
    
control_data = np.empty((len(flies_control),len(statchoice)))    
for i,f in enumerate(flies_control):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    xdat = op.extract_stats_threshold(df,700,statchoice)
    plt.scatter(np.arange(0,len(xdat))+0.25+np.random.rand(len(xdat))*0.05,xdat,color='k')
    control_data[i,:] = xdat


#%% plot data and do stats
from scipy import stats
plt.close('all')
ylabels = ['return distance (mm)','return time (s)','returns per metre', 'plume distance (mm)','distance (mm)' ]
for i in range(len(statchoice)):
    plt.figure(i,figsize=(5,10))
    y_control = control_data[:,i]
    
    x_control = np.ones(y_control.shape)-0.25+np.random.randn(len(y_control))*0.05
    
    y_test = test_data[:,i]
    x_test = np.ones(y_test.shape)+0.25+np.random.randn(len(y_test))*0.05
    
    plt.scatter(x_control,y_control,color='k')
    plt.scatter(x_test,y_test,color='r')
    plt.title(statchoice[i])
    u1,p = stats.mannwhitneyu(y_test[~np.isnan(y_test)],y_control[~np.isnan(y_control)])
    pr = ug.round_to_sig_figs(p,2)
    plt.text(1,np.max(y_test[~np.isnan(y_test)]),'p: ' + str(pr))
    plt.ylabel(ylabels[i])
    plt.xticks([0.75,1.25],labels=['w[118]/UAS-CSChrimson','FB6H (SS95649)/UAS-CSChrimson'],rotation=45)
    plt.subplots_adjust(bottom=0.4,left=0.4)
    plt.xlim([0.5,1.5])
    plt.savefig(os.path.join(savedir,'Blanket_'+statchoice[i]+'.pdf'))
    plt.savefig(os.path.join(savedir,'Blanket_'+statchoice[i]+'.png'))
    
#%% return distance simple
i = 4
plt.figure(figsize=(5,10))
motifdx = [1.1,1,0.1,0]
motifs = ['+','o','+','o']
y_control = control_data[:,i]
x_control = np.ones(y_control.shape)-0.25+np.random.randn(len(y_control))*0.05
dx_control = control_data[:,i+1]

y_test = test_data[:,i]
x_test = np.ones(y_test.shape)+0.25+np.random.randn(len(y_test))*0.05
dx_test = test_data[:,i+1]

for im,m in enumerate(motifdx):
    dx = dx_control==m
    plt.scatter(x_control[dx],y_control[dx],color='k',marker=motifs[im])
    dx = dx_test==m
    plt.scatter(x_test[dx],y_test[dx],color='r',marker=motifs[im])
plt.yscale('log')


plt.title(statchoice[i])
u1,p = stats.mannwhitneyu(y_test[~np.isnan(y_test)],y_control[~np.isnan(y_control)])
pr = ug.round_to_sig_figs(p,2)
plt.text(1,np.max(y_test[~np.isnan(y_test)]),'p: ' + str(pr))
plt.ylabel(ylabels[i])
plt.xticks([0.75,1.25],labels=['w[118]/UAS-CSChrimson','FB6H (SS95649)/UAS-CSChrimson'],rotation=45)
plt.subplots_adjust(bottom=0.4,left=0.4)
plt.xlim([0.5,1.5])
plt.savefig(os.path.join(savedir,'Blanket_FirstReturnSimple.pdf'))
plt.savefig(os.path.join(savedir,'Blanket_FirstReturnSimple.png'))

i = 4
plt.figure(figsize=(5,10))
motifdx = [1.1,1,0.1,0]
motifs = ['x','<','+','>']
y_control = control_data[:,i]
x_control = np.ones(y_control.shape)-0.25+np.random.randn(len(y_control))*0.05
dx_control = control_data[:,i+1]

y_test = test_data[:,i]
x_test = np.ones(y_test.shape)+0.25+np.random.randn(len(y_test))*0.05
dx_test = test_data[:,i+1]

for im,m in enumerate(motifdx):
    dx = dx_control==m
    plt.scatter(x_control[dx],y_control[dx],color='k',marker=motifs[im])
    dx = dx_test==m
    plt.scatter(x_test[dx],y_test[dx],color='r',marker=motifs[im])


plt.title(statchoice[i])
excldx = np.in1d(dx_test,[0.1,1.1])
excldx_c = np.in1d(dx_control,[0.1,1.1])
u1,p = stats.mannwhitneyu(y_test[~excldx],y_control[~excldx_c])
pr = ug.round_to_sig_figs(p,2)
plt.text(1,np.max(y_test[~np.isnan(y_test)]),'p: ' + str(pr))
plt.ylabel(ylabels[i])
plt.xticks([0.75,1.25],labels=['w[118]/UAS-CSChrimson','FB6H (SS95649)/UAS-CSChrimson'],rotation=45)
plt.subplots_adjust(bottom=0.4,left=0.4)
plt.xlim([0.5,1.5])
plt.yscale('log')
plt.savefig(os.path.join(savedir,'Blanket_FirstReturnDetailed.pdf'))
plt.savefig(os.path.join(savedir,'Blanket_FirstReturnDetailed.png'))
