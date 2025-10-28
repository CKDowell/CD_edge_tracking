# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:55:08 2024

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

for i in [1]:
    datadir =os.path.join("Y:\\Data\\FCI\\Hedwig\\hDeltaC_iGluSNFR8880\\241126\\f2\\Trial"+str(i))
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
    #"Y:\\Data\\FCI\\Hedwig\\hDeltaC_iGluSNFR8880\\241126\\f1\\Trial2", #volumetric not many entries
    "Y:\\Data\\FCI\\Hedwig\\hDeltaC_iGluSNFR8880\\241126\\f1\\Trial3", # Single plane lots of entries and jumps
    "Y:\\Data\\FCI\\Hedwig\\hDeltaC_iGluSNFR8880\\241126\\f2\\Trial1",#Volumetric some plume entries
    ]
#regions = ['fsb_whole','fsb_upper']
regions = ['fsb_upper','fsb_lower']
# regions = ['fsb_upper_whole','fsb_lower_whole','fsb_upper','fsb_lower']
for e in experiment_dirs:
    datadir =os.path.join(e)
    print(e)
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    
    cx = CX(name,regions,datadir)
    
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois(dynamicbaseline=True)
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing(uperiod=0.02)
    pv2, ft, ft2, ix = cx.load_postprocessing()
    cxa = CX_a(datadir,regions,yoking=False)
    cxa.save_phases()
    #cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper'],yeseb=False,yk='eb')
    


cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yeseb=False,yk='eb')
#%%
cxa = CX_a(experiment_dirs[0],regions=['fsb_upper'],yoking=False,denovo=False)
cxt = CX_tan(experiment_dirs[1],tnstring='0_fsb_whole')
plt.figure()
plt.plot(cxt.ca)
plt.plot(cxt.ft2['instrip'])
#%%
for i in range(16):
    cxa.pv2[str(i) + '_fsb_upper'] = sg.savgol_filter(cxa.pv2[str(i) + '_fsb_upper'],30,4)
#%%
plt.close('all')
cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper'],yeseb=False,yk='eb')
x = np.linspace(0,len(cxt.ca)+1,len(cxt.ca))
c = sg.savgol_filter(cxt.ca,30,4)
plt.plot(c*10+32.5,x)
plt.figure()
plt.plot(cxt.pv2['relative_time'],cxt.ft2['instrip'])


#plt.plot(cxt.pv2['relative_time'],cxt.ca)
plt.plot(cxt.pv2['relative_time'],c)
#%%
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