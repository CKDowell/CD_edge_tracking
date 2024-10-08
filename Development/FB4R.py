# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:47:27 2024

@author: dowel
"""

from analysis_funs.regression import fci_regmodel
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_tan import CX_tan
import numpy as np
#%% 



for i in [3]:
    datadir =os.path.join("Y:\Data\FCI\\Hedwig\\SS61645_FB4R\\240911\\f1\\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()

#%% ROI processing
for i in [3]:
    datadir =os.path.join("Y:\Data\FCI\\Hedwig\\SS61645_FB4R\\240911\\f1\\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cx = CX(name,['fsbTN'],datadir)
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois()
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()


#pv2, ft, ft2, ix = cx.load_postprocessing()
#%% Plot example data
#datadir = 
datadirs = ["Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240828\\f3\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240910\\f1\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61645_FB4R\\240911\\f1\\Trial3" # Only 2 jumps
            ]
for d in datadirs:
    cxt = CX_tan(d) 
    
    #cxt.fc.example_trajectory_jump(cmin=-0.4,cmax =0.4) 
    plt.figure()
    cxt.fc.mean_traj_nF_jump(cxt.fc.ca,plotjumps=True)

#%%
fc = fci_regmodel(pv2[['0_fsbtn']].to_numpy().flatten(),ft2,pv2)
fc.rebaseline(span=500,plotfig=True)
#%%
y = fc.ca
plt.figure()
plt.plot(y)
plt.plot(ft2['instrip'],color='k')

fc = fci_regmodel(y,ft2,pv2)
fc.example_trajectory(cmin=-0.2,cmax=0.2)
#%%
regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','translational vel','ramp down since exit','ramp to entry']
fc.run(regchoice)
fc.run_dR2(20,fc.xft)


plt.figure()
plt.plot(fc.dR2_mean)
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.xlabel('Regressor name')
plt.show()

plt.figure()
plt.plot(fc.coeff_cv[:-1])
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('Coefficient weight')
plt.xlabel('Regressor name')
plt.show()

plt.figure()
plt.plot([0,len(regchoice)],[0, 0],color='k',linestyle='--') 
plt.plot(-fc.dR2_mean*np.sign(fc.coeff_cv[:-1]),color='k')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2 * sign(coeffs)')
plt.xlabel('Regressor name')
plt.show()