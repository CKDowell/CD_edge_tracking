# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:53:44 2025

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
from Utilities.utils_general import utils_general as ug
#%%
datadirs = [r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial4", # Training changed phase, but could not do post plume
    r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f1\Trial2", # Did post training plume but phase not consistently flipped
            r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250718\f2\Trial4",
            r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial3",#Tracks plume underside of left hand plume, but phase seems to match behaviour
            ]
all_flies = {}
for i,datadir in enumerate(datadirs):
    print(datadir)
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    all_flies.update({str(i):cxa})
    cxa.simple_raw_plot(plotphase=False,regions = ['fsb_upper','fsb_lower'],yk='eb')


#%% 
plt.close('all')
plt.figure()
cxa = all_flies['3']
cxa.mean_phase_train(trng=1)
plt.xlim([-1,61])
#plt.savefig(os.path.join(savedir,'SummaryTrainingDot' + cxa.name + '.pdf'))
#cxa.plot_train_arrow_mean(eb=ebs[i],arrowhead=False,anum=7)
#plt.savefig(os.path.join(savedir,'SummaryTraining' + cxa.name + '.pdf'))
cxa.plot_train_v(plumewidth=30,tperiod = 0.5,plumeang=22.5)
