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
# SS6029 inhibition
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
savedirs = ["Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS6029\\240209\\f2\Trial1",
            "Y:\Data\Optogenetics\PFL3\PFL3_inhibition_SS6029\\240209\\f3\Trial1"
        ]
for i in range(len(savedirs)):
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)