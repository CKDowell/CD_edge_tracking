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

import numpy as np
#%% 



datadir =os.path.join("Y:\Data\FCI\Hedwig\\SS61645_FB4R\\240312\\f1\\Trial2")
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
#%% Registration
ex = im.fly(name, datadir)
ex.register_all_images(overwrite=True)
ex.z_projection()
#%% Masks for ROI drawing
ex.mask_slice = {'All': [1,2,3,4]}
ex.t_projection_mask_slice()

#%% 
cx = CX(name,['fsbTN'],datadir)
# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()
pv2, ft, ft2, ix = cx.load_postprocessing()
#%%
fc = fci_regmodel(pv2[['0_fsbtn']].to_numpy().flatten(),ft2,pv2)
fc.rebaseline(span=500,plotfig=True)
#%%
y = fc.ca
plt.plot(y)
plt.plot(ft2['instrip'],color='k')

fc = fci_regmodel(y,ft2,pv2)
fc.example_trajectory(cmin=0,cmax=0.5)
#%%
regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','x pos','x neg','y pos', 'y neg','ramp down since exit','ramp to entry']
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