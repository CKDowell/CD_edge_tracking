# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:01:30 2024

@author: dowel
"""

from analysis_funs.regression import fci_regmodel
import os
import matplotlib.pyplot as plt 
from analysis_funs.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_tan import CX_tan
import pandas as pd
import numpy as np
#%% Image registration

for i in [1,2,3,4]:
    datadir =os.path.join("Y:\\Data\FCI\\Hedwig\\FB5AB_SS53640\\241205\\f2\\Trial"+str(i))
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
datadirs = [
    "Y:\\Data\FCI\\Hedwig\\FB5AB_SS53640\\241205\\f2\\Trial1",# Not ET
    "Y:\\Data\FCI\\Hedwig\\FB5AB_SS53640\\241205\\f2\\Trial2",# Not ET
    "Y:\\Data\FCI\\Hedwig\\FB5AB_SS53640\\241205\\f2\\Trial3",#ET
    "Y:\\Data\FCI\\Hedwig\\FB5AB_SS53640\\241205\\f2\\Trial4" # Odour pulses
    ]
for datadir in datadirs:

    cx = CX(name,['fsbTN'],datadir)
    # Process ROIs and saves csv
    cx.process_rois()
    
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()
    pv2, ft, ft2, ix = cx.load_postprocessing()




#%% Plume
plt.close('all')
for d in datadirs[:-1]:
    cxt = CX_tan(d)
    cxt.reinit_fc()
    cxt.fc.example_trajectory(cmin=0,cmax=1)
    cxt.fc.example_trajectory_jump(cxt.ca_no_nan.copy(),cxt.ft,cmin=0,cmax=1,cmap='magma')
    plt.figure()
    plt.plot(cxt.ca_no_nan)
    plt.plot(cxt.ft2['instrip'])
    #cxt.fc.mean_traj_nF_jump(cxt.fc.ca,plotjumps=True,cmx=False,offsets=20)
#%% Pulses
cxt = CX_tan(datadirs[0])
#%%
plt.plot(cxt.pv2['0_fsbtn'])
plt.plot(cxt.ft2['instrip'])
plt.plot(cxt.ft2['net_motion'])
plt.figure()
cxt.fc.example_trajectory_scatter(cmin=-.5,cmax=.5)
#%% Regression modelling
regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','angular velocity neg',
                                #'angular acceleration neg','angular acceleration pos',
                                #'angular velocity smooth pos','angular velocity smooth neg',
                                'translational vel','ramp down since exit','ramp to entry']
cxt.fc.run(regchoice)
cxt.fc.run_dR2(20,cxt.fc.xft)

cxt.fc.plot_mean_flur('odour_onset')
cxt.fc.plot_example_flur()
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
#%% Andy
#%%
flies =['Y:\\Data\\FCI\\AndyData\\21D07\\20220103_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220107_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220110_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220111_21D07_sytjGCaMP7f_Fly1_001\\processed']
#%%


datadir = 'Y:\\Data\\FCI\\AndyData\\21D07\\20220111_21D07_sytjGCaMP7f_Fly1_001\\processed'
for datadir in flies:
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    post_processing_file = os.path.join(datadir,'postprocessing.h5')
    pv2 = pd.read_hdf(post_processing_file, 'pv2')
    ft2 = pd.read_hdf(post_processing_file, 'ft2')
    fc = fci_regmodel(pv2['fb5ab_dff'],ft2,pv2)
    fc.example_trajectory(cmin=-0.2,cmax=0.2)
    plt.figure()
    plt.plot(pv2['fb5ab_dff'])
    plt.plot(ft2['instrip'])