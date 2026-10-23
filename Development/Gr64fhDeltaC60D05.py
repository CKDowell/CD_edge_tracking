# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:55:15 2026

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
plt.rcParams['pdf.fonttype'] = 42 
import pingouin as pg
regions = ['eb' ,'fsb_upper','fsb_lower']

#%%
experiment_dirs = [
  
          #r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260824\f1\Trial1', #LED pulses
         # r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260824\f1\Trial2', # Pulse odour medley, animal not walking much. Activity does look lower after GRN stim
            
          # r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260901\f1\Trial1',
          #   r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260901\f1\Trial2',
          
          # r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260911\f1\Trial1', # Pulses
          # r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260911\f1\Trial2', # Pulses
          # r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260911\f1\Trial3' # Version of ET assay
          
          r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260911\f2\Trial1',
          r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260911\f2\Trial2',
          
          
          # r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260914\f1\Trial2', # Pulses
          # r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260914\f1\Trial3' # Pulses
            ]
regions = ['eb' ,'fsb_upper','fsb_lower']
for e in experiment_dirs:
    datadir =os.path.join(e)
    print(e)
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    
    cx = CX(name,regions,datadir)
    
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois()
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()
    pv2, ft, ft2, ix = cx.load_postprocessing()
    cxa = CX_a(datadir,regions=regions,yoking=True,stim=True)
    
    
    cxa.save_phases()
    cxa.simple_raw_plot(regions=['fsb_upper','fsb_lower'],plotphase=True)
    
    
#%% Check data


#datadir = r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260824\f1\Trial2'
datadir = r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260901\f1\Trial2'
datadir = r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260911\f1\Trial1'
datadir = r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260911\f2\Trial2'
cxa = CX_a(datadir,regions=regions,yoking=True,stim=True,denovo=False)
#plt.close('all')
amp = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
#amp = ug.get_pvas(cxa.pdat['wedges_fsb_upper'])
cxa.simple_raw_plot(regions=['eb','fsb_upper','fsb_lower'],plotphase=False)

plt.figure()
ins = cxa.ft2['instrip'].to_numpy()>0
led = cxa.ft2['led1_stpt'].to_numpy()==0
odour = np.logical_and(ins,~led)
ledsin = np.logical_and(led,~ins)
both = np.logical_and(ins,led)
t = np.arange(0,len(amp))/10
plt.plot(t,amp)
plt.plot(t,odour)
plt.plot(t,both)
plt.plot(t,ledsin)
x = cxa.ft2['ft_posx'].to_numpy();y = cxa.ft2['ft_posx'].to_numpy()
u = ug()
_,_,dd = u.get_velocity( x, y, t)
plt.plot(t[1:],-1+dd/20 ,color='k')


timint = 300

ob,obs = ug.find_blocks(odour)
odata = np.zeros((timint,len(ob)))*np.nan
for i,o in enumerate(ob):
    dx = np.arange(o,min(o+timint,len(amp)))
    dxmn = np.arange(o-5,o)
    odata[:len(dx),i] = amp[dx] #- np.mean(amp[dxmn])
    
ob,obs = ug.find_blocks(ledsin)
ldata = np.zeros((timint,len(ob)))*np.nan
for i,o in enumerate(ob):
    dx = np.arange(o,min(o+timint,len(amp)))
    dxmn = np.arange(o-5,o)
    ldata[:len(dx),i] = amp[dx] #- np.mean(amp[dxmn])
    
    
ob,obs = ug.find_blocks(both)
bdata = np.zeros((timint,len(ob)))*np.nan
for i,o in enumerate(ob):
    dx = np.arange(o,min(o+timint,len(amp)))
    dxmn = np.arange(o-5,o)
    bdata[:len(dx),i] = amp[dx]#- np.mean(amp[dxmn])
    
    
omn = np.nanmean(odata,axis=1)
lmn = np.nanmean(ldata,axis=1)
bmn = np.nanmean(bdata,axis=1)
plt.figure()
t = np.arange(0,timint)/10;t = t-2
plt.plot(t,omn,color='k',label='ACV')
plt.plot(t,lmn,color='r',label='LED')
plt.plot(t,bmn,color='m',label='ACV + LED')
plt.fill([0,-2,-2,0],[-.5,-.5,.5,.5],color=[.8,.8,.8],linewidth=0)
plt.legend()
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 5)
cxa.plot_traj_arrow_heat(['fsb_upper'], amp,cmin=0,cmax=.6,colormap='inferno')

# %% Iterate through examples
datadirs = [r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260824\f1\Trial2',
            r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260901\f1\Trial2',
            r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260911\f1\Trial2',
            r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260911\f2\Trial2',
            r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260914\f1\Trial2',
            
            ]

timint = 300
omn_all = np.zeros((timint,len(datadirs)))
lmn_all = np.zeros((timint,len(datadirs)))
bmn_all = np.zeros((timint,len(datadirs)))
pre = np.zeros((len(datadirs),3))
post = np.zeros((len(datadirs),3))
for i, d in enumerate(datadirs):
    cxa = CX_a(d,regions=regions,yoking=True,stim=True,denovo=False)
    amp = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
    ins = cxa.ft2['instrip'].to_numpy()>0
    led = cxa.ft2['led1_stpt'].to_numpy()==0
    odour = np.logical_and(ins,~led)
    ledsin = np.logical_and(led,~ins)
    both = np.logical_and(ins,led)

    ob,obs = ug.find_blocks(odour)
    odata = np.zeros((timint,len(ob)))*np.nan
    pre_all = np.zeros(len(ob))*np.nan
    post_all = np.zeros(len(ob))*np.nan
    for j,o in enumerate(ob):
        dx = np.arange(o,min(o+timint,len(amp)))
        odata[:len(dx),j] = amp[dx]
        pre_all[j] = np.mean(amp[o-1:o])
        offdx = np.arange(o+obs[j]+45,o+obs[j]+55)
        post_all[j] = np.mean(amp[offdx])

    pre[i,0] = np.nanmean(pre_all)
    post[i,0] = np.mean(post_all)


    ob,obs = ug.find_blocks(ledsin)
    ldata = np.zeros((timint,len(ob)))*np.nan
    pre_all = np.zeros(len(ob))*np.nan
    post_all = np.zeros(len(ob))*np.nan
    for j,o in enumerate(ob):
        dx = np.arange(o,min(o+timint,len(amp)))
        ldata[:len(dx),j] = amp[dx]
        pre_all[j] = np.mean(amp[o-1:o])
        offdx = np.arange(o+obs[j]+45,o+obs[j]+55)
        post_all[j] = np.mean(amp[offdx])

    pre[i,1] = np.nanmean(pre_all)
    post[i,1] = np.mean(post_all)


    ob,obs = ug.find_blocks(both)
    bdata = np.zeros((timint,len(ob)))*np.nan
    pre_all = np.zeros(len(ob))*np.nan
    post_all = np.zeros(len(ob))*np.nan
    for j,o in enumerate(ob):
        dx = np.arange(o,min(o+timint,len(amp)))
        bdata[:len(dx),j] = amp[dx]
        pre_all[j] = np.mean(amp[o-1:o])
        offdx = np.arange(o+obs[j]+45,o+obs[j]+55)
        post_all[j] = np.mean(amp[offdx])

    pre[i,2] = np.nanmean(pre_all)
    post[i,2] = np.mean(post_all)

    omn_all[:,i] = np.nanmean(odata,axis=1)
    lmn_all[:,i] = np.nanmean(ldata,axis=1)
    bmn_all[:,i] = np.nanmean(bdata,axis=1)

#%% Stimulus averaged mean across animals
plt.figure()
t = np.arange(0,timint)/10; t = t-2
for i in range(len(datadirs)):
    plt.plot(t,omn_all[:,i],color='k',alpha=0.3)
    plt.plot(t,lmn_all[:,i],color='r',alpha=0.3)
    plt.plot(t,bmn_all[:,i],color='m',alpha=0.3)

plt.plot(t,np.nanmean(omn_all,axis=1),color='k',linewidth=2,label='ACV')
plt.plot(t,np.nanmean(lmn_all,axis=1),color='r',linewidth=2,label='LED')
plt.plot(t,np.nanmean(bmn_all,axis=1),color='m',linewidth=2,label='ACV + LED')
plt.fill([0,-2,-2,0],[-0.1,-0.1,.5,.5],color=[.8,.8,.8],linewidth=0)
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('dF/F0')
savedir = r'Y:\Data\FCI\FCI_summaries\GR64f_combos'
plt.xlim([-2,25])
plt.savefig(os.path.join(savedir,'hDeltaC_Pulse_summary.png'))
plt.savefig(os.path.join(savedir,'hDeltaC_Pulse_summary.pdf'))
#%% Comparison of mean fluorescence 5 seconds after stimulus
plt.close('all')
plt.figure(figsize=(2,6))
x = np.arange(0,3)[:,np.newaxis]
xT = np.tile(x,len(pre))
plt.plot(xT,post.T,color='k',alpha=.3)
colours = ['k','r','m']
for i in range(3):
    plt.scatter(xT[i,:],post[:,i],color=colours[i],zorder=5,s=50)
    
    
df = pd.DataFrame({
    'value': post.T.flatten(),  # Group C
    'group': ['1']*5 + ['2']*5 + ['3']*5
})
aov = pg.anova(df,dv='value',between='group')
posthoc = pg.pairwise_tukey(dv='value', between='group', data=df)

y = np.array([.425,.425])
for i in range(len(posthoc)):
    tdat = posthoc.iloc[i,:]
    x = np.array([int(tdat.A),int(tdat.B)])-1
    plt.plot(x,y,color='k')
    p = ug.round_to_sig_figs(tdat.p_tukey,2)
    plt.text(np.mean(x),y[0]+.005,str(p),horizontalalignment='center',fontsize=12)
    y+=.025

plt.ylabel('dF/F0 10 s post stimulus',fontsize=15)
plt.ylim([0,y[0]])
plt.yticks(np.arange(0,.6,.2),fontsize=15)
plt.xticks([0,1,2],labels=['ACV','LED','ACV + LED'],fontsize=15)
plt.pause(.1)
plt.show()
plt.savefig(os.path.join(savedir,'hDeltaC_Pulse_summary_10s.pdf'))
plt.savefig(os.path.join(savedir,'hDeltaC_Pulse_summary_10s.png'))
#%% Comparison of mean fluorescence 5 seconds after stimulus
x = np.arange(0,3)[:,np.newaxis]
X = np.append(x,x+.5,axis=1)
colours = ['k','r','m']
df = pd.DataFrame({
    'value': post.T.flatten(),  # Group C
    'group': ['1']*5 + ['2']*5 + ['3']*5
})
aov = pg.anova(df,dv='value',between='group')
posthoc = pg.pairwise_tukey(dv='value', between='group', data=df)
ps = posthoc['p_tukey']
for i in range(3):
    y1 = pre[:,i][:,np.newaxis]
    y2 = post[:,i][:,np.newaxis]
    yp = np.append(y1,y2,axis=1)
    x = np.zeros((len(datadirs),2))+i
    x[:,1] = x[:,1]+.5
    plt.plot(x.T,yp.T,color = colours[i],alpha=.5)
    plt.plot(x[0,:],np.mean(yp,axis=0),color=colours[i],linewidth=3)
    

plt.plot([.5,2.5],[.45,.45],color='k')
p = ug.round_to_sig_figs(posthoc['p_tukey'][1],2)
plt.text(1.5,0.47,str(p),horizontalalignment='center')
plt.ylim([0,.5])


#%%
for d in range(len(datadirs)):
    y = pre[d,:][:,np.newaxis]
    y2 = post[d,:][:,np.newaxis]
    yp = np.append(y,y2,axis=1)
    plt.plot(X.T,yp.T,color='k')    


