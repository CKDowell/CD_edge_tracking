# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 09:37:19 2025

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
#%% Image registraion

for i in [1,2,3,4]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250821\f2\Trial"+str(i))
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
    #r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250811\f1\Trial4",
                   # r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250813\f1\Trial1",
                   # r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250813\f1\Trial2",
                   # r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250813\f1\Trial3",
                   # r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250813\f1\Trial4"
                   
                  # r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250818\f2\Trial1", # Central bump like part, but probably motionr artefact
                  # r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250818\f2\Trial2" # SNR too poor to see any bump
                   
                   # r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250818\f1\Trial1",
                   # r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250818\f1\Trial2",
                   # r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250818\f1\Trial3"
                   r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250821\f2\Trial1",
                   r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250821\f2\Trial2",
                   
                   ]
regions = ['fsb']
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
    cx.save_postprocessing()#upsample to 50Hz
    pv2, ft, ft2, ix = cx.load_postprocessing()
    cxa = CX_a(datadir,regions=regions,yoking=False)
    
    cxa.save_phases()
    
#%% Co with pb
experiment_dirs = [
                   r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250821\f2\Trial3",
                   #r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250821\f2\Trial4",
                   
                   ]
regions = ['pb','fsb']
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
    cx.save_postprocessing()#upsample to 50Hz
    pv2, ft, ft2, ix = cx.load_postprocessing()
    cxa = CX_a(datadir,regions=regions,yoking=True,delta7=True)
    
    cxa.save_phases()
#%%

#%% 
datadir= r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250821\f2\Trial1"
cxa = CX_a(datadir,regions=regions,yoking=False,denovo=False)

cxa.simple_raw_plot_new(regions=regions)
#%%
plt.close('all')
datadirs = [r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250813\f1\Trial1",
r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250813\f1\Trial2",
r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250813\f1\Trial3",
r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250813\f1\Trial4",

r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250818\f2\Trial1", 
r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250818\f2\Trial2",

r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250818\f1\Trial1",
r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250818\f1\Trial2",
r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250818\f1\Trial3",
r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250821\f2\Trial2",
]
regions = ['fsb']
for d in datadirs:
    cxa = CX_a(d,regions=regions,yoking=False,denovo=False)
    phase = cxa.pdat['phase_fsb']
    heading = cxa.ft2['ft_heading'].to_numpy()
    ins = cxa.ft2['instrip'].to_numpy()
    t = np.arange(0,len(heading))/10
    try:
        cxa.simple_raw_plot(plotphase=True,yeseb=False)
    except:
        x=1

    # plt.figure()
    # plt.plot(t,heading,color='k')
    # plt.plot(t,ins*np.pi,color='r')
    # #plt.scatter(t,phase,color='b',s=2)
    # # psmooth = ug.boxcar_circ(phase,20)
    # # plt.plot(t,psmooth,color='g')
    # psmooth = ug.savgol_circ(phase,20,3)
    # plt.scatter(t,psmooth,color='g',s=2)
    
    
    
    
#%% Normal tangential post processing ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

experiment_dirs = [
                    r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250821\f2\Trial1",
                   #r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250821\f2\Trial2", # Tracked multiple jumps
                   
                   ]
regions = ['fsbTN']
for e in experiment_dirs:

    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cx = CX(name,['fsbTN'],e)
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois()
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()

#%%
from analysis_funs.CX_analysis_tan import CX_tan
#%%
datadir = r"Y:\Data\FCI\Hedwig\FB6A_SS95731\250821\f2\Trial2"
cxt = CX_tan(datadir) 

#%%
regchoice = ['odour onset', 'odour offset', 'in odour', 'first odour',
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos',
                                #'translational vel dirs',
                                'translational vel',
                                'ramp down since exit','ramp to entry']
#regchoice = ['each odour','ramp to entry']
plt.close('all')
fc =cxt.fc
fc.run(regchoice)
fc.run_dR2(20,fc.xft)
plt.figure(1)
plt.plot(fc.dR2_mean)
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.xlabel('Regressor name')
plt.show()

plt.figure(2)
plt.plot(fc.coeff_cv[:-1])
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('Coefficient weight')
plt.xlabel('Regressor name')
plt.show()

plt.figure(3)
plt.plot([0,len(regchoice)],[0, 0],color='k',linestyle='--') 
plt.plot(-fc.dR2_mean*np.sign(fc.coeff_cv[:-1]),color='k')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2 * sign(coeffs)')
fc.plot_example_flur()
plt.xlabel('Regressor name')
plt.show()











