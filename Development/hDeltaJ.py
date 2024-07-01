# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:16:41 2024

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

for i in [3,2,4]:
    datadir =os.path.join("Y:\\Data\\FCI\\Hedwig\\hDeltaJ\\240529\\f1\\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
#%% Basic data processing
experiment_dirs = [
 
                   "Y:\\Data\\FCI\\Hedwig\\hDeltaJ\\240529\\f1\\Trial3"
                   ]
for e in experiment_dirs:
    datadir =os.path.join(e)
    print(e)
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cx = CX(name,['fsb_lower','fsb_upper','eb'],datadir)
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois()
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()
    pv2, ft, ft2, ix = cx.load_postprocessing()

    try :
        cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'])
    except:
        cxa = CX_a(datadir,regions=['eb','fsb'])
    
    cxa.save_phases()
    
#%% Data exploration
plt.close('all')
cxa.simple_raw_plot(plotphase=True,regions = ['fsb_upper','fsb_lower'])
#cxa.simple_raw_plot(plotphase=True)
#%% 
cxa.point2point_heat(3500,4500,toffset=0,arrowpoint=np.array([50,243,500,600,700,800,900]))
cxa.point2point_heat(0,1000,toffset=0,arrowpoint=np.array([50,243,500,600,700,800,900]))
#%% regression for hDj
regchoice = ['odour onset', 'odour offset', 'in odour', 
                             'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
  'angular velocity pos','angular velocity neg','translational vel','ramp down since exit','ramp to entry']

dr2mat = np.zeros((len(experiment_dirs),len(regchoice)))
dr2mat_max = np.zeros((len(experiment_dirs),len(regchoice)))
savedir = "Y:\\Data\\FCI\\FCI_summaries\\hDeltaJ"
angles = np.linspace(-np.pi,np.pi,16)
for ir, datadir in enumerate(experiment_dirs):
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    
    #y = cxa.pdat['amp_fsb_upper']
    weds = np.sum(cxa.pdat['fit_wedges_fsb_upper']*np.sin(angles),axis=1)
    wedc = np.sum(cxa.pdat['fit_wedges_fsb_upper']*np.cos(angles),axis=1)
    y  = np.sqrt(weds**2+wedc**2)
    ft2 = cxa.ft2
    pv2 = cxa.pv2
    fc = fci_regmodel(y,ft2,pv2)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    dr2mat[ir,:] = (-fc.dR2_mean)*np.sign(fc.coeff_cv[:-1])
    
    y = np.mean(cxa.pdat['wedges_offset_fsb_upper'],axis=1)
    ft2 = cxa.ft2
    pv2 = cxa.pv2
    fc = fci_regmodel(y,ft2,pv2)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    dr2mat_max[ir,:] = (-fc.dR2_mean)*np.sign(fc.coeff_cv[:-1])
    
    fc.plot_example_flur()
    plt.title('Fly: ' + str(ir) +  ' R2:' +str(fc.cvR2))
    plt.savefig(os.path.join(savedir,'EgFit_' + str(ir)+ '.png'))
    plt.figure()
    plt.title(str(ir))
    fc.plot_flur_w_regressors(['in odour','translational vel'],cacol= 'r')
    plt.savefig(os.path.join(savedir,'Ca_withreg_' + str(ir)+ '.png'))
    
#%% 
plt.figure()
x = np.arange(0,len(regchoice))
plt.plot([0,len(regchoice)],[0,0],linestyle='--',color='k')
plt.plot(x,np.transpose(dr2mat),color='k')

plt.xticks(x,labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.title('PVA amplitude regression')
plt.ylabel('Signed dR2')
plt.savefig(os.path.join(savedir,'Reg_Bump_PVA_dR2.png'))

plt.figure()
x = np.arange(0,len(regchoice))
plt.plot([0,len(regchoice)],[0,0],linestyle='--',color='k')
plt.plot(x,np.transpose(dr2mat_max),color='k')
plt.xticks(x,labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.title('Max columns')
plt.ylabel('Signed dR2')
plt.savefig(os.path.join(savedir,'Reg_Bump_Max_dR2.png'))