# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:34:06 2024

@author: dowel
"""

from analysis_funs.regression import fci_regmodel

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
from src.utilities import funcs as fn
plt.rcParams['pdf.fonttype'] = 42 
from Utils.utils_general import utils_general as ug

#%% Image registraion

for i in [3,4,5]:
    datadir =os.path.join("Y:\\Data\\FCI\\Hedwig\\FC2_maimon2\\250129\\f1\\Trial"+str(i))
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
    #"Y:\\Data\\FCI\\Hedwig\\FC2_maimon2\\240821\\f2\\Trial3"
    #"Y:\\Data\\FCI\\Hedwig\\FC2_maimon2\\240916\\f1\\Trial3", # Switch from one to another but not enoughdata
    #"Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial5",# eb good swithc
    #r"Y:\Data\FCI\Hedwig\FC2_maimon2\250124\f1\Trial4",# Training - suspend primed, 1.5 orientations
    #r"Y:\Data\FCI\Hedwig\FC2_maimon2\250124\f1\Trial5",# Training - suspend primed, 1 orientation
    r"Y:\Data\FCI\Hedwig\FC2_maimon2\250220\f1\Trial2"# Training - normal, 3 plumes
                   ]
regions = ['fsb_lower','fsb_upper','eb']
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
    cxa = CX_a(datadir,regions=np.flipud(regions))
    
    
    cxa.save_phases()
#%%
strip_widths = [20,30] # This is for closed loop experiments
strip_angles = [22.5,22.5]
datadir ="Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial5"
datadir = r"Y:\Data\FCI\Hedwig\FC2_maimon2\250220\f1\Trial2"

#datadir = r"Y:\Data\FCI\Hedwig\FC2_maimon2\250124\f1\Trial4"# Training - suspend primed, 1.5 orientations
#datadir = r"Y:\Data\FCI\Hedwig\FC2_maimon2\250124\f1\Trial5"# Training - suspend primed, 1 orientation

#datadir = r"Y:\Data\FCI\Hedwig\FC2_maimon2\250128\f1\Trial2" # Training - could not do second orientation
#datadir =  r"Y:\Data\FCI\Hedwig\FC2_maimon2\250128\f1\Trial3" # Training - did part of first plume

#datadir = r"Y:\Data\FCI\Hedwig\FC2_maimon2\250130\f1\Trial2" # Training - 2.5 plumes NB data corrupted half way thru - unusable :(
#datadir = r"Y:\Data\FCI\Hedwig\FC2_maimon2\250130\f1\Trial3" # Training - powered thru first plume

#datadir = r"Y:\Data\FCI\Hedwig\FC2_maimon2\250130\f2\Trial2" # Training - half a plume


cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False,suspend=False)

plt.plot(cxa.phase_eb)
plt.plot(cxa.ft2['ft_heading'])
plt.plot(cxa.ft2['instrip'])
#plt.plot(cxa.ft2_presuspend['ft_heading'])
#%%
plt.close('all')
cxa.simple_raw_plot(plotphase=True,regions = ['fsb_upper','fsb_lower'],yeseb = True,yk='eb')
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),cxa.pdat['amp_fsb_upper']*2,a_sep=2.5)
cxa.plot_train_v(plumewidth=30,tperiod = 0.5,plumeang=22.5)
plt.figure()
cxa.mean_phase_train(trng=1)

cxa.plot_train_arrow_mean()
#%% Get summary of phase and heading for test and train epochs
train = ft2['intrain']
train[train.isnull()] = 0
train[train==False] = 0
plt.plot(train)

odour = ft2['mfc2_stpt']
instrip = ft2['instrip']
#%% 
r = np.random.random(10)**10
x = r*np.cos(np.pi*22.5/180)
y = r*np.sin(np.pi*22.5/180)

plt.scatter(x,y)
plt.scatter(np.mean(x),np.mean(y))

#%% Good trainers - operant
datadirs =[

    "Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial5"]
ebs = [
       'eb']

savedir = r'Y:\Data\FCI\FCI_summaries\FC2_maimon2_training'
for i,d in enumerate(datadirs):
    
    cxa = CX_a(d,regions=[ebs[i],'fsb_upper','fsb_lower'],denovo=True,suspend=False)
    cxa.plot_train_arrow_mean(eb=ebs[i],arrowhead=False,anum=7)
    plt.savefig(os.path.join(savedir,'SummaryTraining' + cxa.name + '.pdf'))
    
    
    
    
    
    
    