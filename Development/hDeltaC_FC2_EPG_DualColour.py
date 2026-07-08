# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:20:37 2026

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
from Utilities.utils_plotting import uplt as uplt
plt.rcParams['pdf.fonttype'] = 42 

    
    
#%%
datadirs=[ 
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260323\f1\Trial1',#1030nm Not amazing behaviour plus shutter problem
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260323\f1\Trial2',#1030nm Not amazing behaviour
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260323\f1\Trial3', # 1020nm Not amazing behaviour
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260323\f1\Trial4' # 1020nm Did not make jumps but did a good number of entries
    
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260326\f1\Trial2', # Ok ish behaviour. poor hDC signal. Recorded at 1030. 1020 is better
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260326\f1\Trial3', # Not great behaviour
         
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260326\f4\Trial1', # Not good behaviour, EPG signal bad
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260326\f4\Trial4', # Not good behaviour EPG signal bad
    
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f1\Trial1',
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f1\Trial2',
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f1\Trial3', # Nice number of entries 7 jumps two of those jumps are from cross overs
    
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f2\Trial1',
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f2\Trial2',
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f2\Trial3', # Not great tracking, some entries and exits, FC2 goal encoding
    
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f3\Trial1', # Early promise but poor tracking in the end
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f3\Trial2', # Poor tracking
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f3\Trial3',
    
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260401\f1\Trial1',# Not great tracking
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260401\f1\Trial2', # Some nice tracking 3 jumps
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260401\f1\Trial3',# 3 jumps, interesting dataset with plume cross overs

    
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260424\f1\Trial1', # Only a few entries. Clean air FC2 pointing with heading
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260424\f1\Trial2', # Interesting data. FC2 and hDeltaC coming in and out of alignment a lot. Not a great tracker but lots of runs
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260424\f1\Trial3', # Not great behaviour
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260424\f1\Trial4', # Not great behaviour

    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260424\f2\Trial1',# Not great behaviour
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260424\f2\Trial2', # Interesting long runs and few plume interactions. FC2 coming in and out of alignment
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260424\f2\Trial3', # poor behaviour
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260424\f2\Trial4', # Walking and circling with not much FC2 hDeltaC alignment

    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260424\f3\Trial1', #OK behaviour, hDeltaC relatively well correlated with FC2
    
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260425\f2\Trial1', # Data issue with shutter
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260425\f2\Trial2', # Not great behaviour, close FC2 and hDeltaC alignment
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260425\f2\Trial3', # Cool dataset (3 consistent jumps). Lots of entries. Several plumes. hDeltaC seems to be a more reliable goal signal...
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260425\f2\Trial4', # Not great behaviour
    
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260426\f1\Trial1', # Excellent behaviour (8 jumps), with some plume cross overs. Excellent goal encoding FC2, much poorer hDeltaC.
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260426\f1\Trial2', # Ok behaviour. hDeltaC in more of a defulat mode
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260426\f1\Trial3', # Not great beahviour
    
    
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260429\f2\Trial1', # Excellent behaviour 16 or so jumps
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260429\f2\Trial2', # Animal charging through plumes
    # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260429\f2\Trial3',
    
   # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260503\f1\Trial1', # 4 jumps, EGP signal is a bit weird
   # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260503\f1\Trial2',
   # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260503\f1\Trial3'
    
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f1\Trial1', # Both pointing back to plume but did not make jumps
  # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f1\Trial2', # 6 jumps and then plume charging where correlation between hdc and fc2 breaks down. v cool
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f1\Trial3',# downwind charging. Interesting.
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f1\Trial4', # 1 entry
    
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f2\Trial1', # Straight walking through plumes
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f2\Trial2', #Staight walking on entry
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f2\Trial3', # one jump nice transition between tracking and charging
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f2\Trial4', # Charging through
    
   
 # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260617\f1\Trial1',
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260617\f1\Trial2', # Anemotaxis all correlated
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260617\f1\Trial3', # animal not walking
   
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260618\f1\Trial1', # Amenotaxis with some downwind bouts. hDC and FC2 highly correlated. Interesting to see if downwind walking correlates with bump amp size
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260618\f1\Trial2', # Amenotaxis, FC2 signal looks like it drops during amenotaxis
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260618\f1\Trial3', # No ET some circular walking
   
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260619\f1\Trial1',# Amenotaxis FC2 and hDeltaC correlated, poor signal
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260619\f1\Trial2', # Amenotaxis FC2 and hDeltaC correlated, poor signal
  #  r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260619\f1\Trial3', # Amenotaxis FC2 and hDeltaC correlated
    
  #r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260623\f1\Trial1', # Only one odour entry and amenotaxis - had high air flow. Potential for decorrelation of hdc and FC2?
 # r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260623\f1\Trial2', # six jumps lovely dataset with lots of 7 shaped returns.
  
 r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f1\Trial1', # Only one entry, FC2 v quiet
 r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f1\Trial2', # 1 jump. FC2 pops in and out of correlation with hDeltaC
r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f1\Trial3', # No jumps but downwind noodling, could be good to look at hDC driving downwind walking
r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f1\Trial4', # FC2 is well correlated with hDeltaC

r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f2\Trial1', # Strong hDeltaC correlation after odour onset, no tracking
 r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f2\Trial2', # Strong hDeltaC correlation after odour, no tracking
r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f2\Trial3',# Strong hDeltaC correlation after odour, no tracking
r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f2\Trial4',# hDeltaC correlation after odour but no walking
 
    
 
    ]

  

for datadir in datadirs:
   
    regions = ['eb','fsb1','fsb2','fsb1_me','fsb2_me']
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    
    cx = CX(name,regions,datadir)
    
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois()
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()#upsample to 50Hz
    pv2, ft, ft2, ix = cx.load_postprocessing()
    #Channel 2 = Green, Channel 1 = red
    regions = regions = ['eb_ch1','fsb1_ch1','fsb1_ch2','fsb2_ch1','fsb2_ch2','fsb1_me_ch1','fsb1_me_ch2','fsb2_me_ch1','fsb2_me_ch2']
    #regions = ['eb_ch1','eb_ch2','fsb_upper_1_ch1','fsb_lower_1_ch1','fsb_upper_2_ch2']
    #regions = ['fsb_upper_ch1','fsb_upper_ch2','fsb_lower_ch1','fsb_lower_ch2']
    cxa = CX_a(datadir,regions=regions,yoking=True)
    cxa.save_phases()
    try:
        cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
    except:
        print('whoops')
#%% Diagnostic heatmaps

regions = ['eb','fsb1','fsb2']
plt.close('all')
datadir =r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260629\f2\Trial4'
#datadir = r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260426\f1\Trial1'
regions2 = ['eb_ch1','fsb1_ch1','fsb2_ch2']
cxa = CX_a(datadir,regions=regions2,yoking=True,denovo=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=False)
cxa.simple_raw_plot(regions=regions2,yeseb=False,plotphase=True)
#%% Trajectories
region1 = "fsb1_ch1"
region2 = "fsb2_ch2"
colours =  np.array([[49,99,125],[81,156,205]])/255
cxa.plot_traj_arrow_new([region2,region1],a_sep=2,colours =colours,ascale=10)
pva = ug.get_pvas(cxa.pdat['wedges_fsb1_ch1'])
cxa.plot_traj_arrow_heat([region1],pva,a_sep=2,colormap='coolwarm',cmin=0,cmax=.15)


#%% Diagnostic plots
plt.close('all')
colours =  np.array([[49,99,125],[81,156,205]])/255

plt.figure()
pva = ug.get_pvas(cxa.pdat['wedges_fsb1_ch1'])
pvaf = ug.get_pvas(cxa.pdat['wedges_fsb2_ch2'])
offset = stats.circmean(cxa.pdat['phase_eb_ch1']-cxa.ft2['ft_heading'].to_numpy())
#fc2 = cxa.pdat['offset_fsb1_ch1_phase'].to_numpy()
#hdc = ug.circ_subtract(cxa.pdat['offset_fsb2_ch2_phase'].to_numpy(),offset)
fc2 = ug.circ_subtract(cxa.pdat['phase_fsb1_ch1'],offset)
hdc = ug.circ_subtract(cxa.pdat['phase_fsb2_ch2'],offset)

hdcmn = ug.wsumphase(hdc, pva, 50, decay_rate=.1)
fc2mn = ug.wsumphase(fc2, pvaf, 50, decay_rate=.1)
all_entries = cxa.get_entries_exits_like_jumps()


#epg = cxa.pdat['offset_eb_ch1_phase'].to_numpy()
epg = ug.circ_subtract(cxa.pdat['phase_eb_ch1'],offset)

fc2 = cxa.pdat['offset_fsb1_ch1_phase'].to_numpy()
epg = cxa.pdat['offset_eb_ch1_phase'].to_numpy()
hdc = cxa.pdat['offset_fsb2_ch2_phase'].to_numpy()


x = np.arange(0,len(fc2))/10
ins = cxa.ft2['instrip'].to_numpy()
plt.plot(x,ins,color='r')
plt.scatter(x,fc2,color=colours[1,:],s=3)
plt.scatter(x,hdc,color=colours[0,:],s=3)
for ie,e in enumerate(all_entries):
    plt.text(x[e[0]],1,str(ie))



#plt.plot(x,hdcmn,color = colours[0,:])
plt.scatter(x,epg,color='k',s=3)
amp = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
ampf = np.mean(cxa.pdat['wedges_fsb1_ch1'],axis=1)
amp2 = ug.get_pvas(cxa.pdat['wedges_fsb2_ch2'])
plt.plot(x,(amp*3)-6,color=colours[0,:])
plt.plot(x,(ampf*3)-6,color=colours[1,:])
plt.plot(x,(amp2*15)-8,color='r')
plt.plot(x,(pva*15)-8,color='b')

u = ug()
dd,_,_ = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),x)
plt.plot(x[1:],dd/10,color=[.5,.5,.5])

plt.figure()
plt.plot(x,ins,color='r')
plt.scatter(x,fc2,color=colours[1,:],s=3)
plt.scatter(x,hdc,color=colours[0,:],s=3)
plt.scatter(x,epg,color='k',s=3)
for ie,e in enumerate(all_entries):
    plt.text(x[e[0]],1,str(ie))
    
pvfc2 = ug.get_pvas(cxa.pdat['wedges_fsb1_ch1'])
xf = np.sin(fc2)*pvfc2
yf = np.cos(fc2)*pvfc2

xc = np.sin(hdc)*amp2*2
yc = np.cos(hdc)*amp2*2

xd = xf-xc
yd = yf-yc

angsub = np.arctan2(xd,yd)
plt.scatter(x,angsub,color='m',s=3)

plt.figure()
plt.plot(x,ins,color='r')
for ie,e in enumerate(all_entries):
    plt.text(x[e[0]],1,str(ie))
    
fc2_sub = ug.circ_subtract(fc2,epg)
fc2_sub2 = ug.circ_subtract(fc2,hdc)
#plt.scatter(x,fc2_sub,color=colours[1,:],s=5)    
plt.scatter(x,fc2_sub2,color=colours[0,:],s=5)    
plt.scatter(x,fc2+np.pi*2+.5,color=colours[1,:],s=5)
# 

adx = np.logical_and(amp2>0.005,amp>0.5)
adx2 = np.logical_and(amp2>0.005,np.logical_and(amp>0.3,amp<0.5))
plt.figure()
plt.subplot(1,2,1)
plt.scatter(fc2,fc2_sub,s=1)
plt.plot([-np.pi,0],[0,np.pi],color='r')
plt.plot([0,np.pi],[-np.pi,0],color='r')
plt.plot([np.pi/2,np.pi/2],[-np.pi,np.pi],color='r')
plt.plot([-np.pi/2,-np.pi/2],[-np.pi,np.pi],color='r')
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='r')
plt.plot([-np.pi,np.pi],[0,0],color='r')
plt.plot([0,0],[-np.pi,np.pi],color='r')

plt.subplot(1,2,2)
plt.scatter(fc2[adx],fc2_sub2[adx],s=1)
plt.scatter(fc2[adx2],fc2_sub2[adx2],s=1,zorder=-1)
plt.plot([-np.pi,0],[0,np.pi],color='r')
plt.plot([0,np.pi],[-np.pi,0],color='r')
plt.plot([np.pi/2,np.pi/2],[-np.pi,np.pi],color='r')
plt.plot([-np.pi/2,-np.pi/2],[-np.pi,np.pi],color='r')
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='r')
plt.plot([-np.pi,np.pi],[0,0],color='r')
plt.plot([0,0],[-np.pi,np.pi],color='r')

#%% PVA and goal analysis
pva_fc2 = ug.get_pvas(cxa.pdat['wedges_fsb1_ch1'])
fc2 = cxa.pdat['phase_fsb1_ch1'].squeeze()
epg = cxa.pdat['phase_eb_ch1'].squeeze()
hdc = cxa.pdat['phase_fsb2_ch2'].squeeze()

f_e_offset = ug.circ_subtract(fc2,epg)
plt.scatter(pva_fc2,np.abs(f_e_offset),s=1)



#%% 2 D histogram of variables
plt.close('all')
plt.figure()
plt.subplot(1,2,1)
amp_win = amp.copy()
amp2_win = amp2.copy()
amp2_win[amp2<np.percentile(amp2,1)] = np.percentile(amp2,1)
amp2_win[amp2>np.percentile(amp2,99)] = np.percentile(amp2,99)
amp_win[amp<np.percentile(amp,1)] = np.percentile(amp,1)
amp_win[amp>np.percentile(amp,99)] = np.percentile(amp,99)

stat,xedges,yedges,_ = stats.binned_statistic_2d(amp_win,amp2_win,np.abs(fc2_sub2),statistic='median',bins=20)
stat = 180*stat/np.pi
plt.imshow(stat.T)
stat,xedges,yedges,_ = stats.binned_statistic_2d(amp_win,amp2_win,np.abs(fc2_sub2),statistic='count',bins=20)
stat[stat==0] = np.nan
plt.subplot(1,2,2)
plt.imshow(stat.T)
def circular_mean(theta):
    """Compute circular mean of angles in radians."""
    return np.angle(np.mean(np.exp(1j * theta)))


plt.figure()
stat,xedges,yedges,_ = stats.binned_statistic_2d(amp_win,amp2_win,hdc,statistic=circular_mean,bins=20)
stat[stat==0] = np.nan
stat = 180*stat/np.pi
plt.imshow(stat.T,cmap='twilight_shifted',vmin=-180,vmax=180)
plt.colorbar()


plt.figure()
stat,xedges,yedges,_ = stats.binned_statistic_2d(amp_win,amp2_win,fc2,statistic=circular_mean,bins=20)
stat = 180*stat/np.pi
plt.imshow(stat.T,cmap='twilight_shifted',vmin=-180,vmax=180)
plt.colorbar()


plt.figure()
ampf_win = ampf.copy()
ampf_win[ampf<np.percentile(ampf,1)] = np.percentile(ampf,1)
ampf_win[ampf>np.percentile(ampf,99)] = np.percentile(ampf,99)
pvfc2_win = pvfc2.copy()
pvfc2_win[pvfc2<np.percentile(pvfc2,1)] = np.percentile(pvfc2,1)
pvfc2_win[pvfc2>np.percentile(pvfc2,99)] = np.percentile(pvfc2,99)


stat,xedges,yedges,_ = stats.binned_statistic_2d(ampf_win,pvfc2_win,fc2,statistic=circular_mean,bins=20)
stat = 180*stat/np.pi
plt.imshow(stat.T,cmap='twilight_shifted',vmin=-180,vmax=180)
plt.colorbar()
#%% 
# simulate vector sum with different gains
angles= np.linspace(-np.pi,np.pi,100)
gains = np.array([.25,.5,.75,1,1.5,2,4,100])
gains = np.flipud(gains)
gains = -1*(np.cos(angles)-1)
hdc_set = -np.pi
xhdc = np.sin(hdc_set)
yhdc = np.cos(hdc_set)
# for g in gains:
all_angs = np.zeros((len(angles),2))
for i,a in enumerate(angles):

    xn = np.sin(a)*gains[i]
    yn = np.cos(a)*gains[i]
    
    tx = xhdc+xn
    ty = yhdc+yn
    tang = np.arctan2(tx,ty)
    all_angs[i,0] = tang
all_angs[:,1] = ug.circ_subtract(all_angs[:,0],hdc_set)
plt.scatter(all_angs[:,0],all_angs[:,1])
plt.ylim([-np.pi,np.pi])
        



#%%
plt.scatter(x,fc2,color=colours[1,:],s=5)
plt.scatter(x,hdc,color=colours[0,:],s=3)
plt.plot(x,(amp*3)-6,color=colours[0,:])
plt.plot(x,(amp2*15)-8,color='r')
#%% Stop and start phase analysis
u =ug()
x = cxa.ft2['ft_posx'].to_numpy()
y  = cxa.ft2['ft_posy'].to_numpy()
t = cxa.pv2['relative_time'].to_numpy()
vx,vy,vd = u.get_velocity(x, y, t)
vd = np.append(0,vd)
plt.plot(t,vd)
stop = vd<.5
stopblocks,bsize = ug.find_blocks(stop,mergeblocks=True,merg_threshold=3)
plt.scatter(t[stop],vd[stop],color='m',zorder=5)
plt.scatter(t[stopblocks],vd[stopblocks],color='r',zorder=10)
ins = cxa.ft2['instrip'].to_numpy()
minsize = 10
stopblocks = stopblocks[bsize>=minsize]
bsize = bsize[bsize>=minsize]

stopoutside = ins[stopblocks]==0 
stopblocks = stopblocks[stopoutside]
bsize = bsize[stopoutside]

jumps = cxa.get_entries_exits_like_jumps(ent_duration=.1)
jump_real = cxa.get_jumps()

fc2 = cxa.pdat['offset_fsb1_ch1_phase'].to_numpy()
hdc = cxa.pdat['offset_fsb2_ch2_phase'].to_numpy()
epg = cxa.pdat['offset_eb_ch1_phase'].to_numpy()
amp_hdc = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
amp_fc2 = np.mean(cxa.pdat['wedges_fsb1_ch1'],axis=1)

r= ug.rowwise_pearson(cxa.pdat['wedges_fsb1_ch1'],cxa.pdat['wedges_fsb2_ch2'])

colours = uplt.columnar_colours()
colours[0,:] = 0
plt.close('all')
all_diffs = np.empty((0,3))
for ib,b in enumerate(stopblocks[:-1]):
    dx_still = np.arange(b,b+bsize[ib])
    dx_prestill = np.arange(b-50,b)
    jminus = jumps[:,0]-(b+bsize[ib])
    jrminus = jump_real-b
    brmin = (jrminus[:,1]*jrminus[:,2])<0
    if np.sum(brmin)>0:
        isjump = True
        
    else:
        isjump = False
    tend = np.min(np.append(jumps[jminus>0,0],stopblocks[ib+1]))
    
    dx_active = np.arange(b+bsize[ib],tend)
    if len(dx_active)<5:
        continue
    
    
    xa = x[dx_active]
    ya = y[dx_active]
    
    xb = x[dx_prestill]
    yb = y[dx_prestill]
    
    xs = x[dx_still]
    ys = y[dx_still]
    
    xb = xb-xa[0]
    yb = yb-ya[0]
    
    
    xa = xa-xa[0]
    ya = ya-ya[0]
    
    
    
    xrange = np.max(np.abs(xa))
    yrange  = np.max(np.abs(xa))
    drange = np.sqrt(xrange**2+yrange**2)
    scale = drange*.1    

    
    plt.figure()
    
    plt.subplot(2,2,1)
    if isjump:
        plt.title('Jump Return')
    plt.plot(xa,ya,color='k',alpha=.5)
    uplt.coloured_line_simple(xa,ya,vd[dx_active],cmin=0,cmax=10,cmap='coolwarm')
    plt.plot(xb,yb,color=[0.5,0.5,0.5],alpha=.5)
    
    
    dxs = dx_still[-5:]
    
    
    fc2_still = stats.circmean(fc2[dxs],low=-np.pi,high=np.pi)
    hdc_still = stats.circmean(hdc[dxs],low=-np.pi,high=np.pi)
    epg_still = stats.circmean(epg[dxs],low=-np.pi,high=np.pi)
    
    
    # Overall direction
    #plt.plot([0,xa[-1]],[0,ya[-1]],color='r')
    
    # First .5 seconds 
    #plt.plot([0,xa[4]],[0,ya[4]],color='g')
    
    
    
    # plot phases before take off
    px = np.sin(epg_still)*scale
    py = np.cos(epg_still)*scale
    
    plt.plot([0,px],[0,py],color='k')
    
    px = np.sin(hdc_still)*scale
    py = np.cos(hdc_still)*scale
    
    plt.plot([0,px],[0,py],color=colours[1,:])
    
    px = np.sin(fc2_still)*scale
    py = np.cos(fc2_still)*scale
    
    plt.plot([0,px],[0,py],color=colours[2,:])
    
    
    for i in range(0,len(dx_active)-np.mod(len(dx_active),5),5):
        dxa = dx_active[np.arange(i,i+5)]
        fc2_act = stats.circmean(fc2[dxa],low=-np.pi,high=np.pi)
        hdc_act = stats.circmean(hdc[dxa],low=-np.pi,high=np.pi)
        epg_act = stats.circmean(epg[dxa],low=-np.pi,high=np.pi)
    
    
        # plot phases before take off
        px = np.sin(epg_act)*scale+xa[i+3]
        py = np.cos(epg_act)*scale+ya[i+3]
        
        plt.plot([xa[i+3],px],[ya[i+3],py],color='k')
        
        px = np.sin(hdc_act)*scale+xa[i+3]
        py = np.cos(hdc_act)*scale+ya[i+3]
        
        plt.plot([xa[i+3],px],[ya[i+3],py],color=colours[1,:])
        
        px = np.sin(fc2_act)*scale+xa[i+3]
        py = np.cos(fc2_act)*scale+ya[i+3]
        
        plt.plot([xa[i+3],px],[ya[i+3],py],color=colours[2,:])
    
    
    
    traj_angle = np.arctan2(xa[-1],ya[-1])
    diffs = np.array([np.abs(ug.circ_subtract(traj_angle,epg_still)),
                      np.abs(ug.circ_subtract(traj_angle,hdc_still)),
                      np.abs(ug.circ_subtract(traj_angle,fc2_still)),
                      
        ])
    diffs = diffs[np.newaxis,:]
    all_diffs = np.append(all_diffs,diffs,axis=0)
    plt.gca().set_aspect('equal')
    plt.subplot(2,2,2)
    plt.plot(dx_still-dx_still[0],amp_hdc[dx_still],color=colours[1,:])
    plt.plot(dx_still-dx_still[0],amp_fc2[dx_still],color=colours[2,:])
    plt.plot(dx_still-dx_still[0],vd[dx_still]/30,color='k')
    
    plt.plot(dx_active-dx_still[0],amp_hdc[dx_active],color=colours[1,:])
    plt.plot(dx_active-dx_still[0],vd[dx_active]/30,color='r')
    plt.plot(dx_active-dx_still[0],amp_fc2[dx_active],color=colours[2,:])
    plt.subplot(2,2,3)
    plt.scatter(dx_still-dx_still[0],fc2[dx_still],s=5,color=colours[2,:])
    plt.scatter(dx_still-dx_still[0],hdc[dx_still],s=5,color=colours[1,:])
    plt.scatter(dx_still-dx_still[0],epg[dx_still],s=5,color=colours[0,:])
    plt.plot([dx_still[-1]-dx_still[0],dx_still[-1]-dx_still[0]],[-np.pi,np.pi],color='r')
    plt.scatter(dx_active-dx_still[0],fc2[dx_active],s=5,color=colours[2,:])
    plt.scatter(dx_active-dx_still[0],hdc[dx_active],s=5,color=colours[1,:])
    plt.scatter(dx_active-dx_still[0],epg[dx_active],s=5,color=colours[0,:])
    plt.ylim([-np.pi,np.pi])
    plt.subplot(2,2,4)
    plt.plot(dx_still-dx_still[0],r[dx_still],color='k')
    plt.plot(dx_still-dx_still[0],r[dx_still]*0,color='k',linestyle='--')
    plt.plot(dx_active-dx_still[0],r[dx_active],color='r')
    plt.plot(dx_active-dx_still[0],r[dx_active]*0,color='k',linestyle='--')
    
    plt.ylim([-1,1])
    
    
bins=np.linspace(0,np.pi,10)
plotbins = bins[1:]-np.mean(np.diff(bins))/2
plt.figure(101)
for i in range(3): 
    thist = np.histogram(all_diffs[:,i],bins=bins)
    plt.plot(plotbins,thist[0],color=colours[i,:])
#%% Correlation between wedges#
plt.close('all')
colours = uplt.columnar_colours()
t = cxa.pv2['relative_time'].to_numpy()
colours[0,:] = 0
ins = cxa.ft2['instrip'].to_numpy()
X = np.ones((len(cxa.pdat['wedges_fsb2_ch2']),16,3))
X[:,:,0] = cxa.pdat['wedges_fsb2_ch2']
X[:,:,1] = cxa.pdat['wedges_eb_ch1']
betas,rsq = ug.rowwise_multivariate(X,cxa.pdat['wedges_fsb1_ch1'])
#betas,rsq = ug.batch_ridge_with_r2(X[:,:,:2], cxa.pdat['wedges_fsb1_ch1'],fit_intercept=True,lam=.1)
betas,rsq = ug.batch_nnls_with_r2(X,cxa.pdat['wedges_fsb1_ch1'])
#betas,rsq = ug.batch_nnls_ridge(X,cxa.pdat['wedges_fsb1_ch1'],lam=.1)
y_hat = (X @ betas[..., None]).squeeze(-1) 

phase_pred = ug.phase_from_wed(y_hat)

offset = stats.circmean(cxa.pdat['phase_eb_ch1']-cxa.ft2['ft_heading'].to_numpy())
#fc2 = cxa.pdat['offset_fsb1_ch1_phase'].to_numpy()
#hdc = ug.circ_subtract(cxa.pdat['offset_fsb2_ch2_phase'].to_numpy(),offset)
fc2 = ug.circ_subtract(cxa.pdat['phase_fsb1_ch1'],offset)
hdc = ug.circ_subtract(cxa.pdat['phase_fsb2_ch2'],offset)
epg = ug.circ_subtract(cxa.pdat['phase_eb_ch1'],offset)
pp = ug.circ_subtract(phase_pred,offset)

r= ug.rowwise_pearson(cxa.pdat['wedges_fsb1_ch1'],cxa.pdat['wedges_fsb2_ch2'])
r2 = ug.rowwise_pearson(cxa.pdat['wedges_fsb1_ch1'],cxa.pdat['wedges_eb_ch1'])

plt.figure()
plt.scatter(t,fc2+7,color=colours[2,:],s=5)
plt.scatter(t,hdc+7,color=colours[1,:],s=5)
plt.scatter(t,pp+7,color='m',s=5)
plt.scatter(t,epg+7,color='k',s=5)
plt.plot(t,ins+7,color='r')
# plt.figure()
# plt.scatter(r,r2,s=1)
# plt.figure()
# plt.plot(r,color=colours[2,:])
# plt.plot(r2,color=colours[0,:])


plt.plot(t,rsq+2)
plt.plot(t[[0,len(r2)-1]],[2,2],color='k')
plt.plot(t[[0,len(r2)-1]],[2.5,2.5],color='k',linestyle='--')
plt.plot(t[[0,len(r2)-1]],[2.25,2.25],color='k',linestyle='--')
plt.plot(t[[0,len(r2)-1]],[3,3],color='k')
plt.plot(t,betas[:,0],color=colours[1,:])
plt.plot(t,betas[:,1],color=colours[0,:])
plt.plot(t,betas[:,2],color=colours[2,:])
plt.plot(t,ins,color='r')

jumps = cxa.get_jumps()
for j in jumps:
    plt.figure()
    dx = np.arange(j[1],j[2])
    plt.plot(rsq[dx]+1)
    plt.plot(betas[dx,0],color=colours[1,:])
    plt.plot(betas[dx,1],color=colours[0,:])
    plt.plot(betas[dx,2],color=colours[2,:])
#%% prediction with time delay
delays = 30
X =np.ones((len(cxa.pdat['wedges_fsb2_ch2'])-delays-1,16,2*delays+1))
betas,rsq = ug.batch_nnls_with_r2(X,cxa.pdat['wedges_fsb1_ch1'])
for d in range(delays):
    X[:,:,d] =  cxa.pdat['wedges_fsb2_ch2'][delays-d:-(d+1),:]
    X[:,:,d+delays] =  cxa.pdat['wedges_eb_ch1'][delays-d:-(d+1),:]

betas,rsq = ug.batch_ridge_with_r2(X[:,:,:30], cxa.pdat['wedges_fsb1_ch1'][:len(X),:],fit_intercept=True,lam=.1)
#%%
plt.subplot(1,2,1)
plt.scatter(hdc,fc2,s=1)
plt.xlabel('hdc')
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='r')
plt.subplot(1,2,2)
plt.scatter(epg,fc2,s=1)
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='r')
plt.xlabel('epg')
#%%
cxa.pdat['offset_eb_phase'] = pd.Series(epg)
cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(fc2)
cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(hdc)
cxa.plot_traj_arrow_new([region2,region1],a_sep=5,colours =colours)

#%%
cxa.mean_jump_arrows(fsb_names=['eb_ch1','fsb1_ch1','fsb2_ch2'])

cxa.mean_jump_lines(fsb_names=['fsb1_ch1','fsb2_ch2'])



#%% Angular difference across returns

# Revise this list with better animals. Also try this analysis with your cytosolic animals
datadirs = [
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260323\f1\Trial4',
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f1\Trial3',
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260426\f1\Trial2',
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260426\f1\Trial1',
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260401\f1\Trial3',]
plt.close('all')
tdata = np.array([])
for d in datadirs:
    cxa = CX_a(d,regions=regions2,yoking=True,denovo=False)

    jumps = cxa.get_entries_exits_like_jumps()
    jlen = jumps[:,2]-jumps[:,1]
    jmin = 5
    jumps = jumps[jlen>jmin,:]
    jlen = jlen[jlen>jmin]
    jmax = 600
    jumps = jumps[jlen<jmax,:]
    jlen = jlen[jlen<jmax]
    
    jrank = np.argsort(jlen)
    fc2 = cxa.pdat['phase_fsb1_ch1']
    hdc = cxa.pdat['phase_fsb2_ch2']
    pdiff = np.abs(ug.circ_subtract(fc2,hdc))
    tdata  = np.full((600,len(jumps)),np.nan)
    tdata2 = np.full((600,len(jumps)),np.nan)
    for j,ij in enumerate(jrank):
        dx = np.arange(jumps[ij,1],jumps[ij,2])
        x = np.arange(0,len(dx))/10
        plt.figure(1)
        plt.scatter(x,pdiff[dx],color='k',s=1,alpha=.1)
        plt.figure(2)
        plt.scatter(x-np.max(x),pdiff[dx],color='k',s=1,alpha=.1)
        tdata[:len(dx),j] = pdiff[dx] 
        tdata2[-len(dx):,j] = pdiff[dx]
    plt.figure(1)
    x = np.arange(0,600)/10
    plt.plot(x,np.nanmedian(tdata,axis=1),color='r')
    
    plt.figure(2)
    x = np.arange(-600,0)/10
    plt.plot(x,np.nanmedian(tdata2,axis=1),color='r')    
plt.figure(1)
plt.ylim([0,np.pi])
plt.yticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
plt.xlim([0,10])

plt.figure(2)
plt.ylim([0,np.pi])
plt.yticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])

plt.xlim([-10,0])


#%% Mutual information between FC2, EPG and hDC during returns
from scipy.ndimage import gaussian_filter
datadir = r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260426\f1\Trial1'

regions2 = ['eb_ch1','fsb1_ch1','fsb2_ch2']

cxa = CX_a(datadir,regions=regions2,yoking=True,denovo=False)
from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper
etp = ET_paper(datadir,regions=regions2)

#%% Sanity check 'good flies'
colours = uplt.columnar_colours()
colours[0,:] = 0
regions2 = ['eb_ch1','fsb1_ch1','fsb2_ch2']

datadirs = [r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f1\Trial3', # 6 jumps
r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260426\f1\Trial1', # 8 jumps
r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260429\f2\Trial1',# 17 jumps
r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260616\f1\Trial2', # 6 jumps
r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260623\f1\Trial2' # 6 jumps
]
for datadir in datadirs:
    etp = ET_paper(datadir,regions=regions2)
    plt.figure()
    etp.plt_tmp(regions =['eb_ch1','fsb2_ch2','fsb1_ch1'],colours=colours,phase_num=20)
#%% Simple mutual information 
plt.close('all')
bin_number = 20
fc2 = cxa.pdat['phase_fsb1_ch1']
epg = cxa.pdat['phase_eb_ch1']
hdc = cxa.pdat['phase_fsb2_ch2']
hdc_a = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
offset = stats.circmean(cxa.pdat['phase_eb_ch1']-cxa.ft2['ft_heading'].to_numpy())

epg = ug.circ_subtract(epg,offset)
fc2 = ug.circ_subtract(fc2,offset)
hdc = ug.circ_subtract(hdc,offset)

etp.cxa.pdat['offset_eb_ch1_phase'] = pd.Series(epg)
etp.cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(fc2)
etp.cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(hdc)

cxa.pdat['offset_eb_ch1_phase'] = pd.Series(epg)
cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(fc2)
cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(hdc)

# Determine pdensities
bins = np.linspace(-np.pi,np.pi,bin_number)
pfc2 = np.histogram(fc2,bins,density=True)[0]
pfc2 = pfc2/np.sum(pfc2)

pepg = np.histogram(epg,bins,density=True)[0]
pepg = pepg/np.sum(pepg)

phdc = np.histogram(hdc,bins,density=True)[0]
phdc = phdc/np.sum(phdc)
plotbins = bins[1:]-np.mean(np.diff(bins))/2
plt.plot(plotbins,phdc,color='r')
plt.plot(plotbins,pfc2,color='b')
plt.plot(plotbins,pepg,color='k')
plt.xlabel('phase (rad)')
plt.ylabel('probability')

pfc2_epg = np.histogram2d(fc2,epg,bins=bins,density=True)[0]
if bin_number>20:
    pfc2_epg = gaussian_filter(pfc2_epg,2)
pfc2_epg = pfc2_epg/np.sum(pfc2_epg)

pfc2_hdc = np.histogram2d(fc2,hdc,bins=bins,density=True)[0]
if bin_number>20:
    pfc2_hdc = gaussian_filter(pfc2_hdc,2)
pfc2_hdc = pfc2_hdc/np.sum(pfc2_hdc)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(pfc2_epg)
plt.subplot(1,2,2)
plt.imshow(pfc2_hdc)

#p1 is fc2
#p2 is epg/hdc
pbins = bins[1:]-np.mean(np.diff(bins))/2
def MI(p1,p2,p1_p2,bins,p1_obvs,p2_obvs):
    p1_oI = ug.find_nearest_block(p1_obvs,bins) # convert observed phase into indices 
    p2_oI = ug.find_nearest_block(p2_obvs,bins)
    
    MI = 0 
    
    for i in range(len(p1_oI)):
        prob1 = p1[p1_oI[i]]
        prob2 = p2[p2_oI[i]]
        prob1_prob2 = p1_p2[p1_oI[i],p2_oI[i]]
        #MI+= prob2*prob1_prob2*np.log2(prob1_prob2/prob1) # Formula for conditional probabilities
        MI+= prob1_prob2*np.log2((prob1_prob2)/(prob1*prob2))
    
    return MI
jumps = cxa.get_jumps()
#jumps = cxa.get_entries_exits_like_jumps()
jlen = jumps[:,2]-jumps[:,1]
jmin = 30 
jmax = 600
jumps = jumps[np.logical_and(jlen>jmin,jlen<jmax),:]
def entropy(p):
    p = p[p>0]
    return -np.sum(p*np.log2(p))


regions2 = ['fsb1_ch1','eb_ch1','fsb2_ch2']
phase,traj,amp = cxa.pseudo_time_data(jumps,bins=50,regions=regions2)
allMI = np.zeros((len(phase),2))

for i in range(2):
    if i==0:
        jointp = pfc2_epg
        psecond = pepg
    elif i==1:
        jointp = pfc2_hdc
        psecond = phdc
        
    for ir in range(len(phase)):
        print(i,ir)
        allMI[ir,i] = MI(pfc2,psecond,jointp,pbins,phase[ir,0,:],phase[ir,i+1,:])/entropy(pfc2)

plt.figure()
plt.plot(allMI)
#%% Mutual information with hdc amplitude

datadirs = [
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260323\f1\Trial4',
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260331\f1\Trial3',
    r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260401\f1\Trial3']
bin_number = 20
timebins = 50
t = np.append(np.linspace(0,1,50),np.linspace(1,4,50))
allMI2= np.zeros((timebins*2,len(datadirs)))
MIflat = np.zeros((timebins*2,len(datadirs)))
for di,d in enumerate(datadirs):
    plt.figure()
    cxa = CX_a(d,regions=regions2,yoking=True,denovo=False)
    
    fc2 = cxa.pdat['phase_fsb1_ch1']
    epg = cxa.pdat['phase_eb_ch1']
    hdc = cxa.pdat['phase_fsb2_ch2']
    hdc_a = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
    
    
    offset = stats.circmean(cxa.pdat['phase_eb_ch1']-cxa.ft2['ft_heading'].to_numpy())
    
    epg = ug.circ_subtract(epg,offset)
    fc2 = ug.circ_subtract(fc2,offset)
    hdc = ug.circ_subtract(hdc,offset)
    
    etp.cxa.pdat['offset_eb_ch1_phase'] = pd.Series(epg)
    etp.cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(fc2)
    etp.cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(hdc)
    
    cxa.pdat['offset_eb_ch1_phase'] = pd.Series(epg)
    cxa.pdat['offset_fsb1_ch1_phase'] = pd.Series(fc2)
    cxa.pdat['offset_fsb2_ch2_phase'] = pd.Series(hdc)
    
    # Determine pdensities
    bins = np.linspace(-np.pi,np.pi,bin_number)
    pfc2 = np.histogram(fc2,bins,density=True)[0]
    pfc2 = pfc2/np.sum(pfc2)
    
    phdc_amp = np.histogram2d(hdc,hdc_a,bins=bin_number-1,density=True)
    ampbins = phdc_amp[2][1:]
    ampbins = ampbins-np.mean(np.diff(ampbins))/2
    phdc_amp = gaussian_filter(phdc_amp[0],2)
    phdc_amp = phdc_amp/np.sum(phdc_amp)
    
    pfc3_hdc_amp = np.histogramdd( np.concatenate((fc2[:,np.newaxis],hdc[:,np.newaxis],hdc_a[:,np.newaxis]),axis=1),bins=bin_number-1,density=True)[0]
    pfc3_hdc_amp = gaussian_filter(pfc3_hdc_amp,2)
    pfc3_hdc_amp = pfc3_hdc_amp/np.sum(pfc3_hdc_amp)
    # phdc = np.histogram(hdc,bins,density=True)[0]
    # phdc = phdc/np.sum(phdc)
    pfc3_hdc_amp_flat = np.ones_like(pfc3_hdc_amp)*np.mean(pfc3_hdc_amp)
    
    
    #p1 is fc2
    #p2 is epg/hdc
    # a is hdc amplitude
    pbins = bins[1:]-np.mean(np.diff(bins))/2
    def MI(p1,p2_a,p1_p2_a,bins,binsa,p1_obvs,p2_obvs,a_obvs):
        p1_oI = ug.find_nearest_block(p1_obvs,bins) # convert observed phase into indices 
        p2_oI = ug.find_nearest_block(p2_obvs,bins)
        a_oI = ug.find_nearest_block(a_obvs,binsa)
        
        MI = 0 
        
        for i in range(len(p1_oI)):
            prob1 = p1[p1_oI[i]]
            prob2 = p2_a[p2_oI[i],a_oI[i]]
            prob1_prob2_a = p1_p2_a[p1_oI[i],p2_oI[i],a_oI[i]]
            #MI+= prob2*prob1_prob2*np.log2(prob1_prob2/prob1) # Formula for conditional probabilities
            MI+= prob1_prob2_a*np.log2((prob1_prob2_a)/(prob1*prob2))
        return MI
    
    cxa.side = -1
    jumps = cxa.get_entries_exits_like_jumps()
    jlen = jumps[:,2]-jumps[:,1]
    jmin = 30 
    jmax = 600
    jumps = jumps[np.logical_and(jlen>jmin,jlen<jmax),:]
    
    
    
    regions2 = ['fsb1_ch1','eb_ch1','fsb2_ch2']
    phase,traj,amp = cxa.pseudo_time_data(jumps,bins=50,regions=regions2)
    
    
            
    for ir in range(len(phase)):
        
        allMI2[ir,di] = MI(pfc2,phdc_amp,pfc3_hdc_amp,pbins,ampbins,phase[ir,0,:],phase[ir,2,:],amp[ir,1,2,:])/entropy(pfc2)
        MIflat[ir,di] = MI(pfc2,phdc_amp,pfc3_hdc_amp_flat,pbins,ampbins,phase[ir,0,:],phase[ir,2,:],amp[ir,1,2,:])/entropy(pfc2)
    #plt.plot(allMI)
plt.plot(t,allMI2,color='k')
plt.plot(t,np.mean(allMI2,axis=1),color='r')
#plt.plot(t,MIflat,color='r')
    
#%% Difference in wedges
datadir = r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260401\f1\Trial3'
cxa = CX_a(datadir,regions=regions2,yoking=True,denovo=False)
fc2w = cxa.pdat['wedges_fsb1_ch1']
hdcw = cxa.pdat['wedges_fsb2_ch2']
fc2w_n = fc2w/np.max(fc2w,axis=1)[:,np.newaxis]
hdcw_n = hdcw/np.max(hdcw,axis=1)[:,np.newaxis]

ins = cxa.ft2['instrip'].to_numpy()
u = ug()
dv,_,_ = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'].to_numpy())

wdiff = np.sum(np.abs(fc2w-hdcw),axis=1)
wdiff = np.sum(np.abs(fc2w_n-hdcw_n),axis=1)
plt.plot(wdiff,color='k')
plt.plot(ins, color='r')
plt.plot(dv/5-2,color=[.5,.5,.5])
#%%
fc2p = ug.get_pvas(fc2w)
hdcp = ug.get_pvas(hdcw)
fc2ph = cxa.pdat['phase_fsb1_ch1']
hdcph = cxa.pdat['phase_fsb2_ch2']
fc2x = np.sin(fc2ph)*fc2p
fc2y = np.cos(fc2ph)*fc2p
fc2xy = np.append(fc2x[:,np.newaxis],fc2y[:,np.newaxis],axis=1)

hdcx = np.sin(hdcph)*hdcp
hdcy = np.cos(hdcph)*hdcp
hdcxy = np.append(hdcx[:,np.newaxis],hdcy[:,np.newaxis],axis=1)

perms = np.arange(0,600)
lags = np.zeros(len(perms))
jumps = cxa.get_entries_exits_like_jumps()
jlen = jumps[:,2]-jumps[:,1]
jmin = 30 
jmax = 600
jumps = jumps[np.logical_and(jlen>jmin,jlen<jmax),:]
retdx = np.array([],dtype=int)
for j in jumps:
    retdx = np.append(retdx,np.arange(j[1],j[2]))


for p in perms:
    tx = fc2xy[p:,:]
    if p>0:
        ty = hdcxy[:-p,:]
    else:
        ty = hdcxy
    tdot = np.sum(tx*ty,axis=1)
    dx = retdx+p
    dx = dx[dx<len(tdot)]
    tmean = np.mean(tdot)
    print(p,tmean)
    lags[p] = tmean
    
    
    
plt.plot(lags)











