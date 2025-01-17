# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:50:42 2024

@author: dowel
"""

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
#%%
plt.close('all')
meta_data = {'stim_type': 'pulse',
              'act_inhib':'inhib',
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

rootdir = "Y:\\Data\\Optogenetics\\Gr5a_LexA\\GR5a_Lex_Chrimson_Chr2"
flies = ["240827\\f1\\Trial1",
         "240827\\f4\\Trial1",
         "240827\\f6\\Trial1",]

for i,f in enumerate(flies):
    searchdir = os.path.join(rootdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)