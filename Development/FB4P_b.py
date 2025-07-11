# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:56:30 2024

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
#%%
for i in [1,2]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\FB4P_b_SS60296\250311\f1\Trial"+str(i))
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
for i in [1,2]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\FB4P_b_SS60296\250311\f1\Trial"+str(i))
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

#%% 
datadirs = [r"Y:\Data\FCI\Hedwig\FB4P_b_SS60296\240912\f2\Trial3",
            r"Y:\Data\FCI\Hedwig\FB4P_b_SS60296\250311\f1\Trial1",# ET Good 5 mm jumps
            r"Y:\Data\FCI\Hedwig\FB4P_b_SS60296\250311\f1\Trial2" #Replay
            ]
datadir = datadirs[1]
#%%
cxt = CX_tan(datadir)
cxt.fc.example_trajectory_jump(cxt.fc.ca,cxt.ft,cmin=-0.5,cmax =0.5,jsize=5) 
cxt.fc.example_trajectory_scatter(cmin=-0.5,cmax=0.5)
savename = os.path.join(datadir , 'Eg_traj'+ name +'.pdf')
plt.savefig(savename)
#%%

cxt.fc.mean_traj_nF_jump(cxt.fc.ca,plotjumps=True)
#%%

fc = fci_regmodel(pv2[['0_fsbtn']].to_numpy().flatten(),ft2,pv2)
fc.rebaseline(span=500,plotfig=True)
#%% 
y = fc.ca


fc = fci_regmodel(y,ft2,pv2)
fc.example_trajectory(cmin=-0.5,cmax=0.5)
plt.figure()
plt.plot(y)
plt.plot(ft2['instrip'],color='k')

#%%
fc = fci_regmodel(cxt.pv2[['0_fsbtn']].to_numpy().flatten(),cxt.ft2,cxt.pv2)
fc.rebaseline(span=500,plotfig=True)
regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','angular velocity neg','translational vel','ramp down since exit','ramp to entry']
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

#%% Columnar check
datadir =os.path.join(r"Y:\Data\FCI\Hedwig\FB4P_b_SS60296\250311\f1\Trial1")
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
cx = CX(name,['fsbTN','fsb'],datadir)
# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()
#%%
from analysis_funs.CX_analysis_col import CX_a
from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper
cxa = CX_a(datadir,regions=['fsb'],yoking=False)
cxa.simple_raw_plot(yeseb=False)

    

cxa.save_phases()
#%%
plt.close('all')
etp = ET_paper(datadir,regions=['fsb'])
#%%
plt.close('all')
tstart = 110
tend = 300
savedir = r'Y:\Presentations\ForVanessa'
etp.example_trajectory(tstart,tend)
plt.savefig(os.path.join(savedir,'FB4P_bTraj.pdf'))
etp.example_heatmaps(tstart,tend,regions=['fsb'],yoking=False,cmaps=['Purples'])
plt.figure(2)
plt.savefig(os.path.join(savedir,'FB4P_bCa.pdf'))
plt.figure(4)
plt.savefig(os.path.join(savedir,'plume.pdf'))


#%% Including syt gCaMP

plt.close('all')
datadirs = [
    "Y:\\Data\\FCI\\Hedwig\\FB4P_b_SS60296\\240912\\f2\\Trial3",#27 jumps
    "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240809\\f2\\Trial2",#24 jumps
     "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240806\\f1\\Trial5",#2 jumps
     "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240805\\f1\\Trial3",# 2 jumps
    # "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240909\\f1\\Trial5",# 1 jump
       "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240906\\f2\\Trial2",# 2 jumps
       r"Y:\Data\FCI\Hedwig\FB4P_b_SS60296\250311\f1\Trial1"
      ]

plt.close('all')

regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'translational vel dirs',
                                ]
reglabels =['odour onset', 'odour offset', 'in odour', 
                                'tv -157.5','tv -112.5', 'tv -67.5', 
                                'tv -22.5', 'tv  22.5',   'tv 67.5',  
                                'tv 112.5', 'tv 157.5'
                                ]


for d in datadirs:
    cxt = CX_tan(d) 
    fc = cxt.fc
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    
    plt.figure(1)
    plt.plot(fc.dR2_mean)
    plt.xticks(np.arange(0,len(reglabels)),labels=reglabels,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.ylabel('delta R2')
    plt.xlabel('Regressor name')
    plt.show()
    
    plt.figure(2)
    plt.plot(fc.coeff_cv[:-1])
    plt.xticks(np.arange(0,len(reglabels)),labels=reglabels,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.ylabel('Coefficient weight')
    plt.xlabel('Regressor name')
    plt.show()
    
    plt.figure(3)
    plt.plot([0,len(regchoice)],[0, 0],color='k',linestyle='--') 
    plt.plot(-fc.dR2_mean*np.sign(fc.coeff_cv[:-1]),color='k')
    plt.xticks(np.arange(0,len(reglabels)),labels=reglabels,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.ylabel('delta R2 * sign(coeffs)')
    fc.plot_example_flur()
    plt.xlabel('Regressor name')
    plt.show()




