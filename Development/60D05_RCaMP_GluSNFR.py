# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 14:07:49 2025

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
plt.rcParams['pdf.fonttype'] = 42 
#%% Image registraion

for i in [1]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\60D05\60D05GluSNFR4.8880\250910\f2\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir,dual_color=True)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
    
    
#%% pb
datadir= r"Y:\Data\FCI\Hedwig\60D05\60D05GluSNFR4.8880\250910\f2\Trial1"

  
regions = ['eb']
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
#%% Test of just glu snfr
datadir= r"Y:\Data\FCI\Hedwig\60D05\60D05GluSNFR4.8880\250910\f2\Trial1"
regions = ['eb']
cxa = CX_a(datadir,regions=regions,yoking=False)

cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=True)
#%%
datadir= r"Y:\Data\FCI\Hedwig\60D05\250725\f1\Trial3"

regions = ['pb']
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


#Channel 2 = Green, Channel 1 = red
regions = ['pb_ch1','pb_ch2']
cxa = CX_a(datadir,regions=regions,yoking=False)

cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=True)
plt.figure()
plt.scatter(cxa.pdat['phase_pb_ch1'],cxa.pdat['phase_pb_ch2'],s=5,color='k')
plt.xticks([-np.pi,0,np.pi],labels=[-180,0,180])
plt.yticks([-np.pi,0,np.pi],labels=[-180,0,180])
plt.xlabel('Phase RCaMP')
plt.ylabel('Phase iGluSnFR4.8880')
plt.savefig(os.path.join(datadir,'RedGreenPhase.png'))
#%% eb
datadir= r"Y:\Data\FCI\Hedwig\60D05\250725\f1\Trial4"

  
regions = ['eb']
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
#Channel 2 = Green, Channel 1 = red
regions = ['eb_ch1','eb_ch2']
cxa = CX_a(datadir,regions=regions,yoking=False)

cxa.simple_raw_plot(regions=regions,yeseb=False,plotphase=True)
