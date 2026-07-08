# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:52:41 2026

@author: dowel
"""

#%%
from Utilities.utils_general import utils_general as ug
import os
import numpy as np
from analysis_funs.regression import fci_regmodel
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_tan import CX_tan
import numpy as np

#%%
rootdir = r'Y:\Data\FCI\NoelleData\MBON21'

exps = os.listdir(rootdir)
plt.close('all')
for exp in exps:
    data = ug.load_pick(os.path.join(rootdir,exp))
    pv2 = data['pv2']
    ft2 = data['ft2']
    ca = pv2['0_mbon21'].to_numpy()
    ins = ft2['instrip'].to_numpy()
    u = ug()
    x = ft2['ft_posx'].to_numpy()
    y = ft2['ft_posy'].to_numpy()
    tt = pv2['relative_time'].to_numpy()
    dx,dy,dd = u.get_velocity(x,y,tt)
    
    
    plt.figure()
    plt.plot(tt,ca,color='k')
    plt.plot(tt,-.25+ins*.2,color='r')
    plt.plot(tt[1:],-1+dd/20,color=[.5,.5,.5])
    plt.title(exp)
    try:
        plt.figure()
        fci = fci_regmodel(ca,ft2,pv2)
        fci.example_trajectory_scatter(ca)
    except :
        plt.figure()