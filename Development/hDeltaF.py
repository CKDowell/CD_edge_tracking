# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 00:00:00 2026

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

#%%
experiment_dirs = [
    "Y:\\Data\\FCI\\Ladybird\\hDeltaF_60D05_Gc7f\\260715\\f2\\Trial1",
                   "Y:\\Data\\FCI\\Ladybird\\hDeltaF_60D05_Gc7f\\260715\\f2\\Trial2",
                   "Y:\\Data\\FCI\\Ladybird\\hDeltaF_60D05_Gc7f\\260715\\f2\\Trial3",
                   "Y:\\Data\\FCI\\Ladybird\\hDeltaF_60D05_Gc7f\\260715\\f2\\Trial4"
                   ]

regions = ['fsb','eb']
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
    cxa = CX_a(datadir,regions=np.flipud(regions))
    plt.figure()
    cxa.simple_raw_plot(plotphase=False,regions = ['fsb'],yk='eb')
    cxa.save_phases()
#%%
plt.close('all')
for e in experiment_dirs:
    cxa = CX_a(e,regions=np.flipud(regions),denovo=False)
    cxa.simple_raw_plot(plotphase=False,regions = ['fsb','eb'],yk='eb')
    cxa.simple_raw_plot(plotphase=True,regions = ['eb','fsb'],yk='eb')
    cxa.plot_traj_arrow(cxa.pdat['offset_fsb_phase'].to_numpy(),cxa.pdat['amp_fsb'],a_sep= 2)
    # plt.figure()
    # cxa.mean_jump_arrows(fsb_names=['fsb_upper'])
    # cxa.mean_jump_lines()
#%%
regions = ['fsb','eb']
datadir = "Y:\\Data\\FCI\\Ladybird\\hDeltaF_60D05_Gc7f\\260715\\f2\\Trial1"
cxa = CX_a(datadir,regions=np.flipud(regions),denovo=False)
cxa.simple_raw_plot(regions=np.flipud(regions),yeseb=False,plotphase=False)
cxa.simple_raw_plot(regions=np.flipud(regions),yeseb=False,plotphase=True)
fmn = np.mean(cxa.pdat['wedges_fsb'],axis=1)
plt.plot(fmn)
ins = cxa.ft2['instrip'].to_numpy()
plt.plot(ins,color='r')
pva = ug.get_pvas(cxa.pdat['wedges_fsb'])

plt.figure()
plt.plot(pva)
plt.plot(ins*.1,color='r')
#%%
