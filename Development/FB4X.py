# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:44:43 2024

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

import numpy as np
#%% Image registration

for i in [1,3]:
    datadir =os.path.join("Y:\\Data\\FCI\\Hedwig\\SS70711_FB4X\\241031\\f1\\Trial"+str(i))
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
    # "Y:\Data\\FCI\\Hedwig\\SS70711_FB4X\\241030\\f3\\Trial1",
    # "Y:\Data\\FCI\\Hedwig\\SS70711_FB4X\\241030\\f3\\Trial2",
    # "Y:\Data\\FCI\\Hedwig\\SS70711_FB4X\\241030\\f3\\Trial3",
    # "Y:\Data\\FCI\\Hedwig\\SS70711_FB4X\\241030\\f3\\Trial4",
    "Y:\\Data\\FCI\\Hedwig\\SS70711_FB4X\\241031\\f1\\Trial1",
    #"Y:\\Data\\FCI\\Hedwig\\SS70711_FB4X\\241031\\f1\\Trial3"
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
    
#%% Jump assessment
datadirs = [
    # "Y:\Data\\FCI\\Hedwig\\SS70711_FB4X\\241030\\f3\\Trial1",
    # "Y:\Data\\FCI\\Hedwig\\SS70711_FB4X\\241030\\f3\\Trial2",
    "Y:\Data\\FCI\\Hedwig\\SS70711_FB4X\\241030\\f3\\Trial3",# Good jumps
    # "Y:\Data\\FCI\\Hedwig\\SS70711_FB4X\\241030\\f3\\Trial4",
    # "Y:\\Data\\FCI\\Hedwig\\SS70711_FB4X\\241031\\f1\\Trial1",
    "Y:\\Data\\FCI\\Hedwig\\SS70711_FB4X\\241031\\f1\\Trial3" # Good jumps
    ]
pw = [5,5,5,5,10,10]
for i,datadir in enumerate(datadirs):
    cxt = CX_tan(datadir)
    cxt.fc.example_trajectory_jump(cmin=-0.4,cmax=0.5,pw=pw[i])
    plt.figure()
    cxt.plot_mean_traj_jump(cmin=-0.5,cmax=0.5)
#%% 
fc = fci_regmodel(pv2[['0_fsbtn']].to_numpy().flatten(),ft2,pv2)
fc.rebaseline(span=500,plotfig=True)
#%%
y = fc.ca
plt.plot(y)
plt.plot(ft2['instrip'],color='k')

fc = fci_regmodel(y,ft2,pv2)
fc.example_trajectory(cmin=-0.5,cmax=0.5)


#%%

datadir = "Y:\\Data\\FCI\\Hedwig\\SS70711_FB4X\\241031\\f1\\Trial3" 
cxt = CX_tan(datadir)
#%%
from scipy import signal
x = ft2['ang_velocity'].to_numpy()
xarray = np.ones(10)/10
x2 = signal.convolve(x,xarray,method="direct")
#fc.rebaseline(span=500,plotfig=True)
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
#cxt.fc.plot_flur_w_regressors(['angular velocity neg','angular velocity pos'],cacol='r')


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
#%% Summarise all
savedir = "Y:\Data\FCI\Hedwig\\SS70711_FB4X\\SummaryFigures"
datadirs = ["Y:\Data\FCI\Hedwig\\SS70711_FB4X\\240307\\f1\\Trial3",
            "Y:\Data\FCI\Hedwig\\SS70711_FB4X\\240313\\f1\\Trial3",
            "Y:\Data\FCI\Hedwig\\SS70711_FB4X\\240531\\f1\\Trial3"]

regchoice = ['odour onset', 'odour offset', 'in odour', 
                                 'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                 'angular velocity pos','angular velocity neg','translational vel','ramp down since exit','ramp to entry']
d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
rsq_t_t = np.zeros((len(datadirs),2))
for i, d in enumerate(datadirs):
    
    dspl = d.split("\\")
    name = dspl[-3] + '_' + dspl[-2] + '_' + dspl[-1]
    cx = CX(name,['fsbTN'],d)
    pv2, ft, ft2, ix = cx.load_postprocessing()
    fc = fci_regmodel(pv2[['0_fsbtn']].to_numpy().flatten(),ft2,pv2)
    fc.rebaseline(span=500,plotfig=True)
    fc.run(regchoice,partition='pre_air')
    fc.run_dR2(20,fc.xft)
    d_R2s[i,:] = fc.dR2_mean
    coeffs[i,:] = fc.coeff_cv[:-1]
    rsq[i] = fc.r2
    rsq_t_t[i,0] = fc.r2_part_train
    rsq_t_t[i,1] = fc.r2_part_test
    
    
fc.plot_mean_flur('odour_onset')
fc.plot_example_flur()

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
plt.savefig(os.path.join(savedir,'Pre_AirTrain.png'))

#%%
datadirs = ["Y:\Data\FCI\Hedwig\\SS70711_FB4X\\240307\\f1\\Trial3",
            "Y:\Data\FCI\Hedwig\\SS70711_FB4X\\240313\\f1\\Trial3",
            "Y:\Data\FCI\Hedwig\\SS70711_FB4X\\240531\\f1\\Trial3"]
for i, d in enumerate(datadirs):
    cxt = CX_tan(d)
    if i>1:
        cxt.fc.example_trajectory_jump(cmin=-0.5,cmax =0.5,xcent=-215)
    else:
            
        cxt.fc.example_trajectory(cmin=-0.5,cmax =0.5)
        