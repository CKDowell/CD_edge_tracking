# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:07:16 2024

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
#%% Imaging 


for i in [1,2,3,4]:
    datadir =os.path.join("Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f3\\Trial"+str(i))
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

datadirs = ["Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240628\\f1\\Trial2",
            "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f1\\Trial1",
            "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f1\\Trial2",
           "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f3\\Trial1",
           "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f3\\Trial2",
           "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f3\\Trial3",
           "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f3\\Trial4",]

for datadir in datadirs:
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
    pv2, ft, ft2, ix = cx.load_postprocessing()

#%%
datadirs = [
   "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240628\\f1\\Trial2",#Nice
            "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f1\\Trial1",#Nice - but worst for this fly
            "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f1\\Trial2",#Best for this fly
          # "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f3\\Trial1",
          # "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f3\\Trial2",
           "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f3\\Trial3",
           ]
plt.close('all')
for d in datadirs:
    cxt = CX_tan(d) 
    
    cxt.fc.example_trajectory_jump(cmin=-0.4,cmax =0.4) 
    #plt.figure()
    #cxt.fc.mean_traj_nF_jump(cxt.fc.ca,plotjumps=True)
#%%
cxt = CX_tan(datadirs[0])
plt.figure()
cxt.fc.example_trajectory_jump(cmin=-0.5,cmax=0.5)
cxt.fc.mean_traj_nF_jump(cxt.fc.ca,plotjumps=True)
#%%
savedir = "Y:\Data\FCI\Hedwig\\FB5I_SS100553\\SummaryFigures"
regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','angular velocity neg','translational vel','ramp down since exit','ramp to entry']
datadirs = ["Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240523\\f1\\Trial4",
    "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240628\\f1\\Trial2"
    ]
d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
rsq_t_t = np.zeros((len(datadirs),2))
for i,d in enumerate(datadirs):
    cxt = CX_tan(d)
    
    #cxt.fc.run(regchoice)
    #cxt.fc.run_dR2(20,cxt.fc.xft)
    
    cxt.fc.rebaseline(span=500,plotfig=True)
    cxt.fc.run(regchoice,partition='pre_air')
    cxt.fc.run_dR2(20,cxt.fc.xft)
    d_R2s[i,:] = cxt.fc.dR2_mean
    coeffs[i,:] = cxt.fc.coeff_cv[:-1]
    rsq[i] = cxt.fc.r2
    rsq_t_t[i,0] = cxt.fc.r2_part_train
    rsq_t_t[i,1] = cxt.fc.r2_part_test
plt.figure()
plt.plot(d_R2s.T,color='k')
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.xlabel('Regressor name')
plt.show()
plt.savefig(os.path.join(savedir,'dR2.png'))

plt.figure()
plt.plot(-d_R2s.T*np.sign(coeffs.T),color='k')
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2* sign(coeffs)')
plt.xlabel('Regressor name')
plt.show()
plt.savefig(os.path.join(savedir,'dR2_mult_coeff.png'))

plt.figure()
plt.plot(coeffs.T,color='k')
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('Coefficient weight')
plt.xlabel('Regressor name')
plt.show()
plt.savefig(os.path.join(savedir,'Coeffs.png'))

plt.figure()
plt.scatter(rsq_t_t[:,0],rsq_t_t[:,1],color='k')
plt.plot([np.min(rsq_t_t[:]),np.max(rsq_t_t[:])], [np.min(rsq_t_t[:]),np.max(rsq_t_t[:])],color='k',linestyle='--' )
plt.xlabel('R2 pre air')
plt.ylabel('R2 live air')
plt.title('Model trained on pre air period')
#%% 
cxt.fc.plot_example_flur()
times = cxt.pv2['relative_time']
plt.plot(times,cxt.ft2['instrip'])
#%% plot multiple animals
datadirs = [
    "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240523\\f1\\Trial4",
    "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240628\\f1\\Trial2"
    ]
for d in datadirs:
    cxt = CX_tan(d)
    plt.figure()
    #cxt.fc.example_trajectory_jump(cmin=-0.5,cmax =0.5)
   # cxt.fc.example_trajectory_jump(cmin=-0.4,cmax =0.4) 
    plt.figure()
    cxt.fc.mean_traj_nF_jump(cxt.fc.ca,plotjumps=True)
    #plt.savefig(os.path.join(d,'EgTraj_'+cxt.name+'.pdf'))
