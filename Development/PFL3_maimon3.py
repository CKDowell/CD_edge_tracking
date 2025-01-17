# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:51:57 2024

@author: dowel
"""

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
from Utils.utils_general import utils_general as ug
plt.rcParams['pdf.fonttype'] = 42 
#%% 
#%% Image registraion

for i in [2,3]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f2\Trial"+str(i))
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
    r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f1\Trial1",#Good behaviour but dim left LAL
    r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f1\Trial2",
    r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f2\Trial1",
    r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f2\Trial2",
    r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f2\Trial3"
                   ]

regions = ['LAL']
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
#%%
datadir = r"Y:\Data\FCI\Hedwig\PFL3_maimon3\241219\f1\Trial1"
cx = CX(name,regions,datadir)
pv2, ft, ft2, ix = cx.load_postprocessing()

#%% 
from analysis_funs.PFL3_analysis import PFL3_analysis as PFL3

p3 = PFL3(experiment_dirs[2])
p3.fit_PFL3()
p3.plot_goal_arrows()
cx = CX(name,regions,experiment_dirs[2])
pv2, ft, ft2, ix = cx.load_postprocessing()


#%%
plt.close('all')
L = p3.pv2['0_lal'].to_numpy()
R = p3.pv2['1_lal'].to_numpy()
L[np.isnan(L)] = 0
R[np.isnan(R)] = 0

Ls = sg.savgol_filter(L,25,5)
Rs = sg.savgol_filter(R,25,5)
ins = p3.ft2['instrip']
plt.plot(L,color='b')
plt.plot(Ls,color='b')
plt.plot(R,color='r')
plt.plot(Rs,color='r')
plt.plot(ins,color='k')
plt.figure()
plt.plot(R-L,color='b')
plt.plot(ins,color='k')
plt.figure()
plt.plot(R+L,color='b')
plt.plot(ins,color='k')
#%% 
from analysis_funs.regression import fci_regmodel
fci = fci_regmodel(R-L,p3.ft2,p3.pv2)
#fci.rebaseline()
fci.example_trajectory_jump(cmin=-0.75,cmax=0.75)

fci = fci_regmodel((R+L)/2,p3.ft2,p3.pv2)
fci.rebaseline()
fci.example_trajectory_jump(cmin=-0.75,cmax=0.75)

#%% Infer goal

def PFL3_function(heading,goal):
    x = np.linspace(-np.pi,np.pi,16)
    xblock = np.ones((len(heading),len(x)))
    xblock = xblock*x.T
    goal = np.tile(goal,[16,1]).T
    heading = np.tile(heading,[16,1]).T
    
    gsignal = np.cos(goal+xblock)
    hsignal_L = np.cos(heading+xblock+np.pi/4)
    hsignal_R = np.cos(heading+xblock-np.pi/4)
    
    PFL3_L = np.exp(gsignal+hsignal_L)-1
    PFL3_R = np.exp(gsignal+hsignal_R)-1
    
    R_Lal = np.sum(PFL3_L,axis=1)/18.44
    L_Lal = np.sum(PFL3_R,axis= 1)/18.44
    df = np.arctan2(R_Lal,L_Lal)
    return df
def PFL3_function2(heading,goal):
    x = np.linspace(-np.pi,np.pi,16)
    xblock = np.ones((len(heading),len(x)))
    xblock = xblock*x.T
    goal = np.tile(goal,[16,1]).T
    heading = np.tile(heading,[16,1]).T
    
    gsignal = np.cos(goal+xblock)
    hsignal_L = np.cos(heading+xblock+np.pi/4)
    hsignal_R = np.cos(heading+xblock-np.pi/4)
    
    PFL3_L = np.exp(gsignal+hsignal_L)-1
    PFL3_R = np.exp(gsignal+hsignal_R)-1
    
    R_Lal = np.sum(PFL3_L,axis=1)/18.44
    L_Lal = np.sum(PFL3_R,axis= 1)/18.44
    return R_Lal,L_Lal

def PFL3_fun_anyl(heading, goal):
    hL = fn.wrap(heading-np.pi/4)
    hR = fn.wrap(heading+np.pi/4)
    gdiff_L = fn.wrap(hL-goal)
    gdiff_R = fn.wrap(hR-goal)
    L_Lal = np.cos(gdiff_L)
    R_Lal = np.cos(gdiff_R)
    return R_Lal,L_Lal

#%%
from scipy.optimize import minimize

def minfun(goal,heading,target):
    predicted = PFL3_function(heading, np.array([goal]))
    return np.sum((predicted - target) ** 2)
def minfun2(goal,heading,RL,LL):
    predicted_R, predicted_L = PFL3_function2(heading, np.array([goal]))
    error_R = np.sum((predicted_R - RL) ** 2)
    error_L = np.sum((predicted_L - LL) ** 2)
    return error_R + error_L

x = np.linspace(-np.pi,np.pi,100)
rL,lL = PFL3_function2(x*0,x)
df = PFL3_function(x*0,x)
tgcheck = np.zeros_like(x)
for i in range(len(x)):
    #result = minimize(minfun, 0, args=([0], [df[i]]), bounds=[(-np.pi, np.pi)])
    result = minimize(minfun2,0,args=([0], [rL[i]], [lL[i]]), bounds=[(-np.pi, np.pi)])
    tgcheck[i] = result.x[0]


plt.scatter(x,tgcheck)
plt.figure()

plt.plot(x,rL-lL,color='k')
rLp,lLp = PFL3_function2(x*0,tgcheck)
plt.plot(x,rLp-lLp,color='r')

#%%
from scipy.optimize import curve_fit
heading = ft2['ft_heading'].to_numpy()
ydat = R-L
ydat[np.isnan(R-L)] = 0
#PFL3_function([0],[-np.pi/2])

tgoal = np.zeros_like(heading)
for i in range(len(tgoal)):
    result = minimize(minfun2,0,args=([heading[i]],[R[i]],[L[i]]),bounds=[(-np.pi,np.pi)])
    tgoal[i] = result.x[0]
    #tgoal[i],pcov = curve_fit(PFL3_function,[heading[i]],[ydat[i]],bounds=(-np.pi,np.pi))
    
    
#%%

plt.figure()
plt.plot(heading,color='k')
plt.plot(ins)
plt.plot(tgoal,color='r',alpha=0.5)
plt.plot(p3.infgoal,color='g')
#%% 
plt.close('all')
x = np.linspace(-np.pi,np.pi,100)
df = PFL3_function(x*0,x)
rL,lL = PFL3_function2(x*0,x)
plt.plot(x,rL-lL)

tgcheck = np.zeros_like(x)
for i in range(len(tgcheck)):
    tgcheck[i],pcov = curve_fit(PFL3_function,[0],[x[i]],bounds=(-np.pi,np.pi),p0=0)

rLc,lLc = PFL3_function2(x*0,tgcheck)
plt.plot(x,rL,color='r')
plt.plot(x,lL,color='b')


plt.scatter(x,rLc-lLc)
plt.figure()
plt.scatter(x,tgcheck)
plt.figure()
plt.scatter(L,R,s=0.1)
plt.scatter(rL,lL,c=df,cmap='coolwarm')
#%%

xnew,ynew,headingnew = cx.bumpstraighten(ft,ft2)
heading = ft2['ft_heading'].to_numpy()
heading2 = ft['ft_heading'].to_numpy()
bump1 = ft2['bump'].to_numpy()
bump2 = ft['bump'].to_numpy()
bump2[np.isnan(bump2)] = 0
tbumps = bump2[np.abs(bump2)>0]
bump1[np.abs(bump1)>0] = tbumps

plt.figure()
plt.plot(heading)
plt.plot(bump1)
plt.plot(headingnew)
plt.figure()
plt.plot(heading2)
plt.plot(bump2)
#%%
plt.plot(ins,color='m')
plt.plot((Rs-Ls)*3,color='r')
#plt.plot(-(Rs-Ls)*3,color='g',linestyle='--')
hvel = np.diff(sg.savgol_filter(fn.unwrap(heading),50,8)*10)

plt.plot(hvel,color='k')
plt.plot(bump1,color='b')

