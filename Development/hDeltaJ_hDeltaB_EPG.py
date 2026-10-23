# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:46:21 2026

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
  # r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\260911\f1\Trial1',
  # r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\260911\f1\Trial2',
  # r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\260911\f1\Trial3'
  
  r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\261020\f1\Trial1', # only one entry and drift towards the end
  r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\261020\f1\Trial2',# lots of circling
 r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\261020\f1\Trial3',# not tracking
  
  
  r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\261020\f2\Trial1', # Not tracking
  
  r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\261020\f3\Trial1', # Charging through plumes
  r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\261020\f3\Trial2', #some plume interactions
 r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\261020\f3\Trial3'
 
    ]

  

for datadir in datadirs:
   
    regions = ['eb','fsb1_lower','fsb2','fsb1_upper']
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
    regions = regions = ['eb_ch2','fsb1_lower_ch1','fsb2_ch2']
    #regions = ['eb_ch1','eb_ch2','fsb_upper_1_ch1','fsb_lower_1_ch1','fsb_upper_2_ch2']
    #regions = ['fsb_upper_ch1','fsb_upper_ch2','fsb_lower_ch1','fsb_lower_ch2']
    cxa = CX_a(datadir,regions=regions,yoking=True)
    cxa.save_phases()
    try:
        cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
    except:
        print('whoops')


#%% 
plt.close('all')
datadir =  r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\261020\f1\Trial2'
colours = uplt.columnar_colours()
regions =  ['eb_ch2','fsb2_ch2','fsb1_lower_ch1']
cxa = CX_a(datadir,regions=regions,yoking=True,denovo=True)
cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=False)
cxa.plot_traj_arrow_new(regions,a_sep=5)
hdb = cxa.pdat['offset_fsb2_ch2_phase'].to_numpy()
hdj = cxa.pdat['offset_fsb1_lower_ch1_phase'].to_numpy()
eb = cxa.pdat['offset_eb_ch2_phase'].to_numpy()
#hdb = cxa.pdat['phase_fsb2_ch2']
#hdj = cxa.pdat['phase_fsb1_lower_ch1']
ins = cxa.ft2['instrip'].to_numpy()
heading = cxa.ft2['ft_heading'].to_numpy()
t = np.arange(0,len(ins))/10

plt.figure()
plt.plot(t,ins,color='r')
plt.scatter(t,hdb,color='k',s=3)
plt.scatter(t,eb,color=[.5,.5,.5],s=3)
plt.scatter(t,hdj,color=colours[0,:],s=3)
plt.plot(t,heading,color=[.5,.5,.5])
plt.figure()
da = np.abs(ug.circ_subtract(hdb,hdj))
plt.plot(t,da,color='k')
plt.plot(t,ins,color='r')

u =  ug()
dx,dy,dd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),t)
plt.plot(t[1:],-2+dd/10,color=[.5,.5,.5])
plt.figure()
ins1 = np.where(ins>0)[0][0]
#plt.scatter(hdb[:ins1],heading[:ins1],color='k',s=5)
hdiff = ug.circ_subtract(eb,hdb)
plt.plot([t[0],t[-1]],[-np.pi/2,-np.pi/2],color='r')
plt.plot([t[0],t[-1]],[np.pi/2,np.pi/2],color='r')
plt.scatter(t,hdiff,color='k',s=3)
plt.plot(t[1:],-5+dd/10,color=[.5,.5,.5])
still = ug.get_still(.5,dd)
still = np.append(0,still)
plt.plot(t,heading,color=[.5,.3,.3])
plt.plot(t,ins,color='r')
plt.figure()
plt.hist(hdiff[still>0],100)


plt.figure()
plt.scatter(cxa.pdat['phase_eb_ch2'],cxa.pdat['phase_fsb2_ch2'],color='k',s=1)
plt.scatter(cxa.pdat['phase_eb_ch2'],cxa.pdat['phase_fsb1_lower_ch1'],color=colours[0,:],s=1)
plt.xlabel('EPG')
plt.ylabel('FSB')
plt.figure()
plt.scatter(cxa.pdat['phase_fsb2_ch2'],cxa.pdat['phase_fsb1_lower_ch1'],color=colours[0,:],s=1)

