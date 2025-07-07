# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:46:53 2025

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

plt.rcParams['pdf.fonttype'] = 42 
#%% Good tracker trials for reference

datadir_good = ["Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial5",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial2",
r"Y:\Data\FCI\Hedwig\FC2_maimon2\250128\f1\Trial4"]

#%% Other trials from these good trackers that were not as good

datadir_ok = [
    r'Y:\Data\FCI\Hedwig\FC2_maimon2\240411\f2\Trial5', # intermittent tracking
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial2", # Bad walker

"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial2", #Bad walker

"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial1", # decent tracker, just did not make jumps

r'Y:\Data\FCI\Hedwig\FC2_maimon2\240514\f1\Trial3', # decent tracker, plume crossing, single plume

r'Y:\Data\FCI\Hedwig\FC2_maimon2\240821\f2\Trial3', # bouts of tracking with long excursions

#r'Y:\Data\FCI\Hedwig\FC2_maimon2\241024\f2\Trial3', # ok on one plume

r'Y:\Data\FCI\Hedwig\FC2_maimon2\241025\f2\Trial1', # fly walked off after some ET
r'Y:\Data\FCI\Hedwig\FC2_maimon2\241025\f2\Trial3', # fly walked off after some ET
r'Y:\Data\FCI\Hedwig\FC2_maimon2\241025\f2\Trial4', # interesting plume crossing, could have some lookback activity
r'Y:\Data\FCI\Hedwig\FC2_maimon2\241025\f2\Trial5', # leave plume

r"Y:\Data\FCI\Hedwig\FC2_maimon2\241029\f2\Trial1", #Not a great tracker or walker, left after a few entries
r'Y:\Data\FCI\Hedwig\FC2_maimon2\241029\f2\Trial2', # Not great tracker, bump seems to be dim... could be interesting

r'Y:\Data\FCI\Hedwig\FC2_maimon2\241030\f1\Trial1', # Some edge tracking but not great
r'Y:\Data\FCI\Hedwig\FC2_maimon2\241030\f1\Trial3',# Not great edge tracking leaves after a few entries

"Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial2",# Long plume entry and one plume
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial3",# Long plume entries and cross overs -  a bit like the hdC
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial4", # loopy returns, phase pointed out of plume

"Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial1",#ET walk off and then lost

r"Y:\Data\FCI\Hedwig\FC2_maimon2\250124\f1\Trial1",# Cross overs and not amazing tracking

r"Y:\Data\FCI\Hedwig\FC2_maimon2\250128\f1\Trial1", # Cross over and lost 

r"Y:\Data\FCI\Hedwig\FC2_maimon2\250130\f1\Trial1", # very few plume interactions, left plume


]
#%%
plt.close('all')
regions = ['eb','fsb_upper','fsb_lower']
for d in datadir_ok:
    cxa = CX_a(d,regions=regions,denovo=False)
    cxa.simple_raw_plot(plotphase=True,regions = ['fsb_upper','fsb_lower'],yk='eb')
    plt.title(cxa.name)
    cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),cxa.pdat['amp_fsb_upper'],a_sep= 5)
    plt.title(cxa.name)
#%% Single tests
plt.close('all')
d = r"Y:\Data\FCI\Hedwig\FC2_maimon2\250130\f2\Trial1"
cxa = CX_a(d,regions=regions,denovo=False)
cxa.simple_raw_plot(plotphase=True,regions = ['fsb_upper','fsb_lower'],yk='eb')
plt.title(cxa.name)
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'].to_numpy(),cxa.pdat['amp_fsb_upper'],a_sep= 2.5)
plt.title(cxa.name)
    
#%% processed raw imaging data
for i in [4,5]:
    
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\FC2_maimon2\241111\f1\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice() 

#%% process unprocessed data
d = r"Y:\Data\FCI\Hedwig\FC2_maimon2\241111\f1\Trial5"

dn = d.split("\\")
name = dn[-3] + '_' + dn[-2] + '_' + dn[-1]

cx = CX(name,regions,d)

# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()#upsample to 50Hz
pv2, ft, ft2, ix = cx.load_postprocessing()
cxa = CX_a(d,regions=regions)

cxa.save_phases()

    




