# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:12:03 2024

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
#%% Image registraion

for i in [1,2,3,4,5]:
    datadir =os.path.join("Y:\Data\FCI\Hedwig\FB4P_b_SS67631_sytGC7f\\240909\\f1\\Trial"+str(i))
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
   # "Y:\Data\FCI\\Hedwig\\FB4P_b_SS67631_sytGC7f\\240809\\f2\\Trial2",
   # "Y:\Data\FCI\\Hedwig\\FB4P_b_SS67631_sytGC7f\\240805\\f1\\Trial3",#decent
    # "Y:\Data\FCI\\Hedwig\\FB4P_b_SS67631_sytGC7f\\240806\\f1\\Trial1",
    # "Y:\Data\FCI\\Hedwig\\FB4P_b_SS67631_sytGC7f\\240806\\f1\\Trial4",
    # "Y:\Data\FCI\\Hedwig\\FB4P_b_SS67631_sytGC7f\\240806\\f1\\Trial6",
    # "Y:\Data\FCI\\Hedwig\\FB4P_b_SS67631_sytGC7f\\240806\\f1\\Trial5"#decent
    
   # "Y:\Data\FCI\\Hedwig\\FB4P_b_SS67631_sytGC7f\\240906\\f2\\Trial2",
    "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240906\\f2\\Trial3",
    "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240906\\f2\\Trial4",
    "Y:\Data\FCI\Hedwig\FB4P_b_SS60296_sytGC7f\\240909\\f1\\Trial1",
    "Y:\Data\FCI\Hedwig\FB4P_b_SS60296_sytGC7f\\240909\\f1\\Trial2",
    "Y:\Data\FCI\Hedwig\FB4P_b_SS60296_sytGC7f\\240909\\f1\\Trial3",
    "Y:\Data\FCI\Hedwig\FB4P_b_SS60296_sytGC7f\\240909\\f1\\Trial4",
    "Y:\Data\FCI\Hedwig\FB4P_b_SS60296_sytGC7f\\240909\\f1\\Trial5"
    
                   ]
for e in experiment_dirs:
    datadir =os.path.join(e)
    print(e)
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cx = CX(name,['fsb'],datadir)
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois()
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()
    pv2, ft, ft2, ix = cx.load_postprocessing()

    # try :
    #     cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'])
    # except:
    #     cxa = CX_a(datadir,regions=['fsb'])
    cxa = CX_a(datadir,regions=['fsb'],yoking=False)
    cxa.save_phases()
#%%
datadirs = [
    "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240809\\f2\\Trial2",
    "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240805\\f1\\Trial3",#decent
    # "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240806\\f1\\Trial1",
    # "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240806\\f1\\Trial4",
    # "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240806\\f1\\Trial6",
     "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240806\\f1\\Trial5"#decent
    
                   ]


cxa = CX_a(experiment_dirs[3],regions=['fsb'],yoking=False,denovo=True)
#%%
plt.close('all')
cxa.simple_raw_plot(plotphase=False,regions = ['fsb'],yeseb=False)
#cxa.simple_raw_plot(plotphase=True)
#%%
plt.close('all')
phase = cxa.pdat['offset_fsb_phase']
amp = np.mean(cxa.pdat['fit_wedges_fsb'],axis=1)/10
amp = fc.ca
cxa.plot_traj_arrow(phase,100*amp**4,a_sep=0.5)
for i in range(10):
    cxa.point2point_heat(i*1000,(i+1)*1000,regions=['eb','fsb'],toffset=0)
    
#%%

pv2 = cxa.pv2
y = np.mean(pv2.filter(regex='fsb'),axis=1)

fc = fci_regmodel(y.to_numpy().flatten(),cxa.ft2,pv2)
fc.rebaseline()
#fc.example_trajectory_jump(cmin=-0.2,cmax=0.2)

regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','angular velocity neg','stationary','translational vel','ramp down since exit','ramp to entry']




fc.run(regchoice,partition='pre_air')
fc.run_dR2(20,fc.xft)

fc.plot_mean_flur('odour_onset')
fc.plot_example_flur()
plt.figure()
plt.plot(fc.dR2_mean)
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.xlabel('Regressor name')
plt.show()

plt.figure()
plt.plot(fc.coeff_cv[:-1])
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('Coefficient weight')
plt.xlabel('Regressor name')
plt.show()

#%%
plt.close('all')
datadirs = [
    "Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240809\\f2\\Trial2",
    "Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240805\\f1\\Trial3",#decent
    # "Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240806\\f1\\Trial1",
    # "Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240806\\f1\\Trial4",
    # "Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240806\\f1\\Trial6",
     "Y:\Data\FCI\\Hedwig\\FB4P_b_sytGC7f\\240806\\f1\\Trial5"#decent
    
                   ]

regchoice = ['odour onset', 'odour offset', 'in odour', 
                                 'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                 'angular velocity pos','angular velocity neg','stationary','translational vel','ramp down since exit','ramp to entry']
d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
rsq_t_t = np.zeros((len(datadirs),2))
for i, d in enumerate(datadirs):
    
    cxa = CX_a(d,regions=['eb','fsb'],denovo=False)
    
    pv2 = cxa.pv2
    y = np.mean(pv2.filter(regex='fsb'),axis=1)

    fc = fci_regmodel(y.to_numpy().flatten(),cxa.ft2,pv2)
    fc.rebaseline()
    fc.run(regchoice,partition='pre_air')
    fc.run_dR2(20,fc.xft)
    d_R2s[i,:] = fc.dR2_mean
    coeffs[i,:] = fc.coeff_cv[:-1]
    rsq[i] = fc.r2
    rsq_t_t[i,0] = fc.r2_part_train
    rsq_t_t[i,1] = fc.r2_part_test
    
    
    fc.plot_mean_flur('odour_onset')
    fc.plot_example_flur()
    fc.example_trajectory_jump(cmin=-0.2,cmax=0.2)

plt.figure()
plt.plot(d_R2s.T,color='k')
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.xlabel('Regressor name')
plt.show()
#plt.savefig(os.path.join(savedir,'dR2.png'))

plt.figure()
plt.plot(-d_R2s.T*np.sign(coeffs.T),color='k')
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2* sign(coeffs)')
plt.xlabel('Regressor name')
plt.show()
#plt.savefig(os.path.join(savedir,'dR2_mult_coeff.png'))

plt.figure()
plt.plot(coeffs.T,color='k')
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('Coefficient weight')
plt.xlabel('Regressor name')
plt.show()
#plt.savefig(os.path.join(savedir,'Coeffs.png'))

plt.figure()
plt.scatter(rsq_t_t[:,0],rsq_t_t[:,1],color='k')
plt.plot([np.min(rsq_t_t[:]),np.max(rsq_t_t[:])], [np.min(rsq_t_t[:]),np.max(rsq_t_t[:])],color='k',linestyle='--' )
plt.xlabel('R2 pre air')
plt.ylabel('R2 live air')
plt.title('Model trained on pre air period')
#plt.savefig(os.path.join(savedir,'Pre_AirTrain.png'))
#%%
off = 0
for d in datadirs:
    cxa = CX_a(d,regions=['eb','fsb'],denovo=False)
    
    pv2 = cxa.pv2
    y = np.mean(pv2.filter(regex='fsb'),axis=1)

    fc = fci_regmodel(y.to_numpy().flatten(),cxa.ft2,pv2)
    fc.rebaseline()
    fc.mean_traj_heat_jump(fc.ca,xoffset=off)
    off = off+30
fc.mean_traj_nF_jump(fc.ca,plotjumps=True,cmx=0.1)


fc.mean_traj_heat_jump(fc.ca)




#%%
x1 = cxa.ft2['ft_posx'].copy()
y1 = cxa.ft2['ft_posy'].copy()
dx = np.diff(x1)
dy = np.diff(y1)
trans_diff = np.sqrt(dx**2+dy**2)
x2 = np.append([0],trans_diff)
xp = np.percentile(np.abs(x2),1)
x = np.zeros_like(x2)
x[np.abs(x2)<xp] = 1