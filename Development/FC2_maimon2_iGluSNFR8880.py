# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:50:39 2024

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
from analysis_funs.CX_analysis_tan import CX_tan

plt.rcParams['pdf.fonttype'] = 42 

#%% Image registraion

for i in [1,2,3,4]:
    datadir =os.path.join("Y:\\Data\\FCI\\Hedwig\\FC2_maimon2iGluSNFR8880\\241112\\f2\\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
    
#%%
experiment_dirs = [
    "Y:\\Data\\FCI\\Hedwig\\FC2_maimon2iGluSNFR8880\\241112\\f2\\Trial1",
                   "Y:\\Data\\FCI\\Hedwig\\FC2_maimon2iGluSNFR8880\\241112\\f2\\Trial2",
                   "Y:\\Data\\FCI\\Hedwig\\FC2_maimon2iGluSNFR8880\\241112\\f2\\Trial3",
                   "Y:\\Data\\FCI\\Hedwig\\FC2_maimon2iGluSNFR8880\\241112\\f2\\Trial4"]
regions = ['fsb_whole','fsb']
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
    cx.save_postprocessing(uperiod=0.02)
    pv2, ft, ft2, ix = cx.load_postprocessing()
    cxa = CX_a(datadir,regions=['fsb'],yoking=False)
    cxa.save_phases()
#%%
cxa = CX_a(experiment_dirs[4],regions=['fsb'],yoking=False,denovo=False)

cxa.simple_raw_plot(plotphase=True,regions = ['fsb'],yeseb=False,yk='eb')
#%% 
for i in range(4):
    print(i)
    cxt = CX_tan(experiment_dirs[i],tnstring='0_fsb_whole')
    print(i)
    plt.figure()
    plt.plot(cxt.ca)
    plt.plot(cxt.ft2['instrip'])
    
#%% 
cxt = CX_tan(experiment_dirs[1],tnstring='0_fsb_whole')


regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','angular velocity neg',
                                #'angular acceleration neg','angular acceleration pos',
                                #'angular velocity smooth pos','angular velocity smooth neg',
                                'translational vel','ramp down since exit','ramp to entry']


cxt.fc.run(regchoice)
cxt.fc.run_dR2(20,cxt.fc.xft)
cxt.fc.rebaseline(span=500,plotfig=True)
cxt.fc.example_trajectory_jump(cmin=-0.5,cmax=0.5)

plt.figure()
plt.plot(cxt.fc.dR2_mean)
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.xlabel('Regressor name')
plt.show()

plt.figure()
plt.plot(cxt.fc.coeff_cv[:-1])
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('Coefficient weight')
plt.xlabel('Regressor name')
plt.show()

plt.figure()
plt.plot(-cxt.fc.dR2_mean*np.sign(cxt.fc.coeff_cv[:-1]))
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('signed delta R2')
plt.xlabel('Regressor name')
plt.show()
#%%


#%% Load tiff and do pca
w = cxa.pdat['wedges_fsb']
w2 = np.array([])
for i in range(8):
    tw = w[:,i*2:(i*2+2)]
    tw = np.mean(tw,axis=1)
    tw= tw[:,np.newaxis]
    if i==0:
        w2= tw
    else:
        w2 = np.append(w2,tw,axis=1)
        
        
        
plt.imshow(w2,aspect='auto',vmax=1,interpolation='none',cmap='Greys')
plt.plot(w2[:,[0,-1]])
c = sg.correlate(w2[:,0],w2[:,-1])
c = np.correlate(w[:,8],w[:,9])/len(w)