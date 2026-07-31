# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 14:30:01 2025

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
from Utilities.utils_plotting import uplt as uplt
from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper

plt.rcParams['pdf.fonttype'] = 42 
#%% Image registraion

for i in [1,2,3]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\hDeltaM_SS59766\250829\f1\Trial"+str(i))
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
    
    #r"Y:\Data\FCI\Hedwig\hDeltaM_SS59766\250828\f1\Trial2",
    #r"Y:\Data\FCI\Hedwig\hDeltaM_SS59766\250828\f1\Trial3", #No ET
     # r"Y:\Data\FCI\Hedwig\hDeltaM_SS59766\250828\f1\Trial4",  
        r"Y:\Data\FCI\Hedwig\hDeltaM_SS59766\250829\f1\Trial2"   
                   ]
regions = ['eb','fsb_upper','fsb_lower']
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
    cxa = CX_a(datadir,regions=regions)
    
    
        
    
    cxa.save_phases()
    
#%%
plt.close('all')
#datadir = r"Y:\Data\FCI\Hedwig\hDeltaM_SS59766\250829\f1\Trial2"
datadir = r"Y:\Data\FCI\Hedwig\hDeltaM_SS59766\250829\f1\Trial2"   
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
#%%
savedir = r'Y:\Data\FCI\FCI_summaries\hDeltaM'
plt.close('all')
cxa.simple_raw_plot(plotphase=False,regions = ['eb','fsb_upper','fsb_lower'],yk='eb')
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
p = ug.circ_subtract(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),np.pi)
cxa.plot_traj_arrow(p,np.mean(cxa.pdat['wedges_fsb_upper']/2,axis=1),a_sep= 2)
#cxa.plot_traj_arrow(cxa.pdat['offset_fsb_lower_phase'].to_numpy(),np.mean(cxa.pdat['wedges_fsb_lower']/2,axis=1),a_sep= 2)
#%%
plt.figure()
t  = np.arange(0,len(cxa.pdat['phase_eb']))/10
plt.scatter(t,cxa.pdat['phase_eb'],color='k',s=2)
plt.scatter(t,cxa.pdat['phase_fsb_upper'],color='b',s=2)
plt.scatter(t,cxa.pdat['phase_fsb_lower'],color='m',s=2)
plt.plot(t,cxa.ft2['instrip']*3,color='r')

plt.figure()
cd = np.abs(ug.circ_subtract(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_eb']))
cd2 = np.abs(ug.circ_subtract(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_fsb_lower']))
plt.plot(t,cd,color='b')
#plt.plot(t,cd2,color='m')
plt.plot(t,cxa.ft2['instrip'],color='r')
plt.figure()
cxa.mean_jump_arrows(fsb_names=['eb','fsb_upper','fsb_lower'],jsize =3,ascale=100)

#%%
plt.figure()
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],linestyle='--',color='r')
plt.plot([-np.pi,0],[0,np.pi],linestyle='--',color='r')
plt.plot([0,np.pi],[-np.pi,0],linestyle='--',color='r')

plt.scatter(cxa.pdat['phase_eb'],cxa.pdat['phase_fsb_upper'],color='k',s=1,alpha=.2)
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.xlabel('EPG phase')
plt.ylabel('FSB upper phase')
plt.savefig(os.path.join(savedir,'FSB_EPG_Phase.png'))

plt.figure()
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],linestyle='--',color='r')
plt.plot([-np.pi,0],[0,np.pi],linestyle='--',color='r')
plt.plot([0,np.pi],[-np.pi,0],linestyle='--',color='r')
plt.scatter(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_fsb_lower'],color='k',s=1,alpha=.2)
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.xlabel('FSB upper phase')
plt.ylabel('FSB lower phase')
plt.savefig(os.path.join(savedir,'FSB_upper_lower_phase.png'))


#%% Plot amplitude and PVA of signal

pva = ug.get_pvas(cxa.pdat['wedges_fsb_upper'])
amp = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
ins =cxa.ft2['instrip']
plt.plot(pva*5,color='b')
plt.plot(amp,color='k')
plt.plot(ins,color='r')

#%% plot hDeltaM jumps
colours = uplt.columnar_colours()
colours[0,:] = 0
regions2 = ['eb','fsb_upper','fsb_lower']

datadirs = [r"Y:\Data\FCI\Hedwig\hDeltaM_SS59766\250829\f1\Trial2"  
]
savedir = r'Y:\Data\FCI\FCI_summaries\hDeltaM'
plt.close('all')
colcul = uplt.columnar_colours()
colours = np.array([[0, 0,0],[59, 88, 161],[59, 88, 161]])/255
colours[2,:] = colcul[2,:]
for datadir in datadirs:
    etp = ET_paper(datadir,regions=regions2)
    plt.figure()
    etp.plt_tmp(regions =['eb','fsb_upper','fsb_lower'],colours=colours,phase_num=6,scalar=7)
    plt.savefig(os.path.join(savedir,'Upper_lower_phase_'+etp.cxa.name+'.pdf'))
