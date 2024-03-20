# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:25:59 2024

@author: dowel
"""

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
#%% 
# SS6029
plt.close('all')
meta_data = {'stim_type': 'plume',
              'act_inhib':'inhib',
    'ledOny': 700,
              'ledOffy':10000,
              'ledOnx': -1000,
              'ledOffx': 1000,
              'LEDoutplume': True,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 0,
            
              }
savedirs = ["Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS60291\\240209\\f2\Trial1", # Phen
            "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS60291\\240209\\f3\Trial1",# Phen
            #"Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS60291\\240304\\f1\Trial1", Not ET
            "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS60291\\240304\\f2\Trial1", # No or subtle phen
            "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS60291\\240304\\f3\Trial1", # Looping trajectories
            "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS60291\\240304\\f4\Trial1",# Got lost after ET for some time
            "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS60291\\240304\\f5\Trial1", # No or subtle phen
            "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS60291\\240304\\f6\Trial1" # Got lost after some time
        ]
figsavedir = "Y:\Data\Optogenetics\\PFL3\\PFL3_inhibition_SS60291\\SummaryFigures"
plt.rcParams['pdf.fonttype'] = 42 
for i in range(len(savedirs)):
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    snames = sdir.split('\\')
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.png'
    plt.savefig(os.path.join(figsavedir,savename))
    
    plt.rcParams['pdf.fonttype'] = 42 
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.pdf'
    plt.savefig(os.path.join(figsavedir,savename))
#%% 
# S82335 inhibition
plt.close('all')
meta_data = {'stim_type': 'plume',
              'act_inhib':'inhib',
    'ledOny': 700,
              'ledOffy':1000,
              'ledOnx': -1000,
              'ledOffx': 1000,
              'LEDoutplume': True,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 0,
            
              }
savedirs = [
            "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS82335\\240227\\f1\\Trial1",
            "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS82335\\240227\\f2\\Trial1",
            "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS82335\\240227\\f3\\Trial1",
            "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS82335\\240227\\f4\\Trial1",
        ]
figsavedir = "Y:\Data\Optogenetics\\PFL3\\PFL3_inhibition_SS82335\\SummaryFigures"
for i in range(len(savedirs)):
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    snames = sdir.split('\\')
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.png'
    plt.savefig(os.path.join(figsavedir,savename))
    
    plt.rcParams['pdf.fonttype'] = 42 
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.pdf'
    plt.savefig(os.path.join(figsavedir,savename))
    