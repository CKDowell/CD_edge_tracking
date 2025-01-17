# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:49:00 2025

@author: dowel
"""

import numpy as np
import pandas as pd
import analysis_funs.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
#%% Horizontal ACV plume and stimulation
plt.close('all')
meta_data = {'stim_type': 'plume',
              'act_inhib':'act',
    'ledOny': -float(50)/2,
              'ledOffy':float(50)/2,
              'ledOnx': 10,
              'ledOffx': 30,
              'LEDoutplume': False,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 90,
              'RepeatInterval':250
              }

savedirs = [r"E:\Data\Optogenetics\Jaycie\P1_test\250117\f1\Trial4",
 r"E:\Data\Optogenetics\Jaycie\P1_test\250117\f1\Trial7",
        
        ]
for i in range(len(savedirs)):
    searchdir = savedirs[i]
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
#%% stimulation and no ACV

meta_data = {'stim_type': 'pulse',
              'act_inhib':'act',
    'ledOny': -float(50)/2,
              'ledOffy':float(50)/2,
              'ledOnx': 10,
              'ledOffx': 30,
              'LEDoutplume': False,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 90,
              'RepeatInterval':250
              }

savedirs = [r"E:\Data\Optogenetics\Jaycie\P1_test\250117\f1\Trial1",
            r"E:\Data\Optogenetics\Jaycie\P1_test\250117\f1\Trial2",
            r"E:\Data\Optogenetics\Jaycie\P1_test\250117\f1\Trial3",
            ]

for i in range(len(savedirs)):
    searchdir = savedirs[i]
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    
    
    
    
#%%

savedirs = [r"E:\Data\Optogenetics\Jaycie\P1_test\250117\f1\Trial5",
            ]

meta_data = {'stim_type': 'plume',
              'act_inhib':'act',
    'ledOny': -float(50)/2,
              'ledOffy':float(50)/2,
              'ledOnx': 10,
              'ledOffx': 30,
              'LEDoutplume': False,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 0,
              'RepeatInterval':250
              }
for i in range(len(savedirs)):
    searchdir = savedirs[i]
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    



