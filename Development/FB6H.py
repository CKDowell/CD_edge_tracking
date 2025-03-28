# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:55:33 2025

@author: dowel
"""

from analysis_funs.regression import fci_regmodel
import os
import matplotlib.pyplot as plt 
from analysis_funs.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_tan import CX_tan
import numpy as np
#%%
for i in [4,5]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    ex = im.fly(name, datadir)
    ex.register_all_images(overwrite=True)
    ex.z_projection()
    #%
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
    

#%% ROI processing
for i in [1,2,3,4,5]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial"+str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cx = CX(name,['fsbTN'],datadir)
    # save preprocessing, consolidates behavioural data
    cx.save_preprocessing()
    # Process ROIs and saves csv
    cx.process_rois()
    # Post processing, saves data as h5
    cx.crop = False
    cx.save_postprocessing()
#%% 
datadirs = [r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250313\f1\Trial1",# walking in circles, not great
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250313\f1\Trial2",# 
            #r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250314\f1\Trial1",# Imaging artefact
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250314\f1\Trial2",# Possible artefact half way through
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250314\f1\Trial3", # Interesting plume interactions - No ET.
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250314\f1\Trial4",
            
            
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial1",#Good behaviour, strong sustained suppression during edge tracking
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial2",# Many plume entries and exits with plume disappearance
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial3",# Replay of exp 1, big artefact
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial4", #ACV pulses v nice slow rebound
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial5", # Octanol pulses
            
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f2\Trial1",# one plume no jumps artefacts
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f2\Trial2",# Several plumes, artefacts
            
            
            
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial1",#Good tracker
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial2",#Decent tracker multiple plumes and strong ramping
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial3",# Strong suppression during replay not great behaviour
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial4", #Octanol pulses
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial5",# ACV pulses, neuron inhibited
            
            
            ]
#%%
plt.close('all')
datadir = r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial2"

cxt = CX_tan(datadir)
cxt.fc.example_trajectory_jump(cxt.ca.copy(),cmin=-0.5,cmax =0.5,jsize=5) 
cxt.fc.example_trajectory_scatter(cxt.ca.copy(),cmin=-0.5,cmax=0.5)

cxt.fc.example_trajectory_jump(cxt.ca_rebase.copy(),cmin=-0.5,cmax =0.5,jsize=5) 
cxt.fc.example_trajectory_scatter(cxt.ca_rebase.copy(),cmin=-0.5,cmax=0.5)


#savename = os.path.join(datadir , 'Eg_traj'+ name +'.pdf')
#plt.savefig(savename)


ft2 = cxt.ft2
plt.figure()
plt.plot(ft2['instrip'],color='r')
plt.plot(ft2['mfc3_stpt']/np.max(ft2['mfc3_stpt']),color='g')
#plt.plot(np.abs(ft2['net_motion'].to_numpy()*10),color='k')
plt.plot(cxt.ca,color=[0.2,0.2,1])

cxt.fc.plot_mean_flur('odour_onset')
#%%
datadirs_check = [
           # r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f2\Trial1",
           # r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f2\Trial2",
            
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250319\f1\Trial1", #Good tracker - artefact in data
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250319\f1\Trial2", # Good tracker - artefact in data
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250319\f1\Trial3",
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250319\f1\Trial4", # Artefact in pulses
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250319\f1\Trial5", # Pre air artefact 
            ]

#%%
plt.close('all')
for datadir in datadirs_check:
    d = datadir.split("\\")
    #plt.figure()
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    
    cxt = CX_tan(datadir)
    try:
        cxt.fc.example_trajectory_jump(cxt.ca.copy(),cmin=0,cmax =.75,jsize=5) 
        cxt.fc.example_trajectory_scatter(cxt.ca.copy(),cmin=0,cmax=0.75)
        plt.title(name)
    except:
        print('not plotting traj')
    #savename = os.path.join(datadir , 'Eg_traj'+ name +'.pdf')
    # plt.savefig(savename)
    
    
    ft2 = cxt.ft2
    plt.figure()
    plt.plot(ft2['instrip'],color='r')
    plt.plot(np.abs(ft2['net_motion'].to_numpy()*10),color='k')
    plt.plot(cxt.ca,color=[0.2,0.2,1])
    plt.title(name)
    # cxt.fc.plot_mean_flur('odour_onset')
    
    
    
#%% Tau fitting function
def tau_fitting(regchoice):
    decay_TCs = np.array([0.3,1,2,4,8,16,32,64,128,256])
    #Short tau
    r2s = np.zeros((len(decay_TCs),len(decay_TCs)))
    for i1,d1 in enumerate(decay_TCs):
        for i2,d2 in enumerate(decay_TCs):
            taus = np.array([[d1,0.01],[d2,0.01]]) # Fit first odour first
            #regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour'],taus)
            cxt.fc.run(regchoice,partition=False,cirftau=taus)
            r2s[i1,i2] = cxt.fc.r2
    i= np.argmax(r2s)
    ri, ci = np.unravel_index(i, r2s.shape)
    ri =ri+1
    ci = ci+1 
    decay_TCs = np.append(decay_TCs,512)
    short_tau = decay_TCs[ri]
    long_tau = decay_TCs[ci]
    short_array = np.linspace(decay_TCs[ri-1],decay_TCs[ri+1],5)
    long_array = np.linspace(decay_TCs[ci-1],decay_TCs[ci+1],5)

    serror = decay_TCs[ri+1]-decay_TCs[ri-1]
    lerror = decay_TCs[ci+1]-decay_TCs[ci-1]
    while serror>0.05 or lerror>0.1:
        r2s = np.zeros((len(short_array),len(long_array)))
        for i1,d1 in enumerate(short_array):
            for i2,d2 in enumerate(long_array):
                taus = np.array([[d1,0.01],[d2,0.01]]) # Fit first odour first
                #regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour'],taus)
                cxt.fc.run(regchoice,partition=False,cirftau=taus)
                r2s[i1,i2] = cxt.fc.r2#cross validated ridge is used, so there is some jitter

        i= np.argmax(r2s)
        ri, ci = np.unravel_index(i, r2s.shape)
        short_tau = short_array[ri]
        long_tau = long_array[ci]
        ri = ri+1
        ci = ci+1
        sdiff = np.mean(np.diff(short_array))
        ldiff = np.mean(np.diff(long_array))
        short_array = np.append(np.append(short_array[0]-sdiff,short_array),short_array[-1]+sdiff)
        long_array = np.append(np.append(long_array[0]-ldiff,long_array),long_array[-1]+ldiff)
        
        serror = short_array[ri+1]-short_array[ri-1]
        lerror = long_array[ci+1]-long_array[ci-1]
        
        
        short_array = np.linspace(short_array[ri-1],short_array[ri+1],5)
        long_array = np.linspace(long_array[ci-1],long_array[ci+1],5)
    return short_tau,long_tau
    
#%% ACV pulse modelling


# ACV
# Regression fit timeconstants to odour pulse onset
datadirs =[ r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial4",
           r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial5",
           ]


for datadir in datadirs:
    cxt = CX_tan(datadir)
    short_tau,long_tau = tau_fitting(['odour onset','first odour'])
    taus = np.array([[short_tau,0.01],[long_tau,0.01],[0.3,0.01],[0.3,0.01],[0.3,0.01]]) # Fit first odour first
    cxt.fc.run(['odour onset','first odour','angular velocity abs','translational vel','stationary'],partition=False,cirftau=taus)
    cxt.fc.plot_example_flur()
    plt.title('R2 '+str(cxt.fc.r2)+' Tau 1: ' +str(short_tau) + ' Tau 2: ' +str(long_tau))

#%% Octanol

datadirs =[ r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial5",
           r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial4",
           ]

for datadir in datadirs:
    cxt = CX_tan(datadir)
    short_tau,long_tau = tau_fitting(['oct onset','first oct'])
    taus = np.array([[short_tau,0.01],[long_tau,0.01],[0.3,0.01],[0.3,0.01],[0.3,0.01]]) # Fit first odour first
    cxt.fc.run(['odour onset','first odour','angular velocity abs','translational vel','stationary'],partition=False,cirftau=taus)
    cxt.fc.plot_example_flur()
    plt.title('R2 '+str(cxt.fc.r2)+' Tau 1: ' +str(short_tau) + ' Tau 2: ' +str(long_tau))

#%% replay experiments
datadirs =[ r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial3",#big artefact
           r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial3",
           ]


for datadir in datadirs:
    cxt = CX_tan(datadir)
    short_tau,long_tau = tau_fitting(['odour onset','first odour'])
    taus = np.array([[short_tau,0.01],[long_tau,0.01],[0.3,0.01],[0.3,0.01],[0.3,0.01]]) # Fit first odour first
    cxt.fc.run(['odour onset','first odour','angular velocity abs','translational vel','stationary'],partition=False,cirftau=taus)
    cxt.fc.plot_example_flur()
    plt.title('R2 '+str(cxt.fc.r2)+' Tau 1: ' +str(short_tau) + ' Tau 2: ' +str(long_tau))


#%% # Run pearson regression to get CIRF estimates
decay_TCs = [0.3,1,2,4,8,16,32,64,100,128,200,256]
cxt.reinit_fc()
for d in decay_TCs:
    cxt.fc.run_pearson(['odour onset'],cirftau =[d,0.01])
    rho = cxt.fc.pearson_rho 
    cxt.fc.run_pearson(['first odour'],cirftau =[d,0.01])
    rho2 = cxt.fc.pearson_rho 
    print('Tau ', d,'Odour onset rho ', rho,'First odour rho ', rho2)
    
    
taus = np.array([[6,0.01],[100,0.01]])
#regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour'],taus)
cxt.fc.run(['odour onset','first odour'],partition=False,cirftau=taus)

cxt.fc.plot_example_flur()
#%% Tau optimisation

decay_TCs = np.array([0.3,1,2,4,8,16,32,64,128,256])



#Short tau
r2s = np.zeros_like(decay_TCs)
for i,d in enumerate(decay_TCs):
    taus = np.array([[d,0.01],[0.3,0.01]]) # Fit first odour first
    #regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour'],taus)
    cxt.fc.run(['odour onset','first odour'],partition=False,cirftau=taus)
    r2s[i] = cxt.fc.r2
    
bt = np.argmax(r2s)
tau_error = decay_TCs[bt+1]-decay_TCs[bt-1]
tau_range = decay_TCs
tau_resolution = 0.05
while tau_error>tau_resolution:
    tau_range = np.linspace(tau_range[bt-1],tau_range[bt+1],5)
    r2s1 = np.zeros_like(tau_range)
    
    for i,t in enumerate(tau_range):
        taus = np.array([[t,0.01],[0.3,0.01]]) # Fit first odour first
        #regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour'],taus)
        cxt.fc.run(['odour onset','first odour'],partition=False,cirftau=taus)
        r2s1[i] = cxt.fc.r2
    
    bt = np.argmax(r2s1)+1
    tdiff = np.mean(np.diff(tau_range))
    tau_range = np.append(np.append(tau_range[0]-tdiff,tau_range),tau_range[-1]+tdiff)
    tau_error = tau_range[bt+1]-tau_range[bt-1]
    print('TE!!!')
    print(tau_error)
short_tau =tau_range[bt]

#Long tau
r2s = np.zeros_like(decay_TCs)
for i,d in enumerate(decay_TCs):
    taus = np.array([[short_tau,0.01],[d,0.01]]) # Fit first odour first
    #regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour'],taus)
    cxt.fc.run(['odour onset','first odour'],partition=False,cirftau=taus)
    r2s[i] = cxt.fc.r2

bt = np.argmax(r2s)
tau_error = decay_TCs[bt+1]-decay_TCs[bt-1]
tau_range = decay_TCs
tau_resolution = 0.1 # resolution of 0.1s
while tau_error>tau_resolution:
    tau_range = np.linspace(tau_range[bt-1],tau_range[bt+1],5)
    r2s1 = np.zeros_like(tau_range)
    
    for i,t in enumerate(tau_range):
        taus = np.array([[0.3,0.01],[t,0.01]]) # Fit first odour first
        #regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour'],taus)
        cxt.fc.run(['odour onset','first odour'],partition=False,cirftau=taus)
        r2s1[i] = cxt.fc.r2
    
    bt = np.argmax(r2s1)+1
    tdiff = np.mean(np.diff(tau_range))
    tau_range = np.append(np.append(tau_range[0]-tdiff,tau_range),tau_range[-1]+tdiff)
    tau_error = tau_range[bt+1]-tau_range[bt-1]
    print('TE!!!')
    print(tau_error)
    
long_tau = tau_range[bt]
#%% Fit both at same time - 
plt.close('all')
decay_TCs = np.array([0.3,1,2,4,8,16,32,64,128,256])
#Short tau
r2s = np.zeros((len(decay_TCs),len(decay_TCs)))
for i1,d1 in enumerate(decay_TCs):
    for i2,d2 in enumerate(decay_TCs):
        taus = np.array([[d1,0.01],[d2,0.01]]) # Fit first odour first
        #regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour'],taus)
        cxt.fc.run(['odour onset','first odour'],partition=False,cirftau=taus)
        r2s[i1,i2] = cxt.fc.r2
plt.imshow(r2s)
i= np.argmax(r2s)
ri, ci = np.unravel_index(i, r2s.shape)
short_tau = decay_TCs[ri]
long_tau = decay_TCs[ci]
short_array = np.linspace(decay_TCs[ri-1],decay_TCs[ri+1],5)
long_array = np.linspace(decay_TCs[ci-1],decay_TCs[ci+1],5)

serror = decay_TCs[ri+1]-decay_TCs[ri-1]
lerror = decay_TCs[ci+1]-decay_TCs[ci-1]
while serror>0.05 or lerror>0.1:
    r2s = np.zeros((len(short_array),len(long_array)))
    for i1,d1 in enumerate(short_array):
        for i2,d2 in enumerate(long_array):
            taus = np.array([[d1,0.01],[d2,0.01]]) # Fit first odour first
            #regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour'],taus)
            cxt.fc.run(['odour onset','first odour'],partition=False,cirftau=taus)
            r2s[i1,i2] = cxt.fc.r2#cross validated ridge is used, so there is some jitter
    plt.figure()
    plt.imshow(r2s)
    
    plt.show()
    
    i= np.argmax(r2s)
    ri, ci = np.unravel_index(i, r2s.shape)
    short_tau = short_array[ri]
    long_tau = long_array[ci]
    ri = ri+1
    ci = ci+1
    sdiff = np.mean(np.diff(short_array))
    ldiff = np.mean(np.diff(long_array))
    short_array = np.append(np.append(short_array[0]-sdiff,short_array),short_array[-1]+sdiff)
    long_array = np.append(np.append(long_array[0]-ldiff,long_array),long_array[-1]+ldiff)
    
    serror = short_array[ri+1]-short_array[ri-1]
    lerror = long_array[ci+1]-long_array[ci-1]
    
    
    short_array = np.linspace(short_array[ri-1],short_array[ri+1],5)
    long_array = np.linspace(long_array[ci-1],long_array[ci+1],5)
    
#%%
taus = np.array([[short_tau,0.01],[long_tau,0.01]]) # Fit first odour first
#regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour'],taus)
cxt.fc.run(['odour onset','first odour'],partition=False,cirftau=taus)
cxt.fc.plot_example_flur()
plt.title('R2 '+str(cxt.fc.r2))

#%% full model
taus = np.array([[short_tau,0.01],[long_tau,0.01],[0.3,0.01],[0.3,0.01],[0.3,0.01]]) # Fit first odour first
#regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour'],taus)
cxt.fc.run(['odour onset','first odour','angular velocity abs','translational vel','stationary'],partition=False,cirftau=taus)
cxt.fc.plot_example_flur()
plt.title('R2 '+str(cxt.fc.r2))
#regmatrix, regmatrix_preconv = cxt.fc.set_up_regressors(['odour onset','first odour','angular velocity abs','translational vel','stationary'],taus)
