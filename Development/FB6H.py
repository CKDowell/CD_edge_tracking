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
from Utilities.utils_general import utils_general as ug
#%%
for i in [2]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250507\f2\Trial"+str(i))
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
for i in [2]:
    datadir =os.path.join(r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250507\f2\Trial"+str(i))
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
            
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250328\f2\Trial1", # One jump, not great tracker
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250328\f2\Trial3",# Made a few jumps, not a great tracker
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250328\f2\Trial4",# Not a great tracker
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250328\f2\Trial5",# ACV two timecourses. Animal walks well
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250328\f2\Trial6",# Octanol pulses. Two timecourses
            
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250507\f2\Trial2" # Made a few jumps, ramps after plume exits
            
            ]
#%%
plt.close('all')
datadir = r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250507\f2\Trial2"

cxt = CX_tan(datadir)
ca = cxt.ca.copy().ravel()
ca[-1] = 0
cxt.fc.example_trajectory_jump(ca,cxt.ft,cmin=-0.5,cmax =0.5,jsize=5) 
cxt.fc.example_trajectory_scatter(cxt.ca.copy(),cmin=-0.5,cmax=0.5)

cxt.fc.example_trajectory_jump(cxt.ca_rebase.copy(),cxt.ft,cmin=-0.5,cmax =0.5,jsize=5) 
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
    decay_TCs = np.append(0.15,decay_TCs,)
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
           r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250328\f2\Trial5"
           ]

ACV_taus = np.zeros((len(datadirs),2))
dR2 = np.zeros((len(datadirs),3))
for i,datadir in enumerate(datadirs):
    cxt = CX_tan(datadir)
    cxt.reinit_fc()
    short_tau,long_tau = tau_fitting(['odour onset','first odour'])
    taus = np.array([[short_tau,0.01],[long_tau,0.01],[0.3,0.01],[0.3,0.01],[0.3,0.01]]) # Fit first odour first
    #cxt.fc.run(['odour onset','first odour','angular velocity abs','translational vel','stationary'],partition=False,cirftau=taus)
    cxt.fc.run(['odour onset','first odour','angular velocity abs'],partition=False,cirftau=taus)
    cxt.fc.plot_example_flur()
    plt.title('R2 '+str(cxt.fc.r2)+' Tau 1: ' +str(short_tau) + ' Tau 2: ' +str(long_tau))
    ACV_taus[i,0] = short_tau
    ACV_taus[i,1] = long_tau
    cxt.fc.run_dR2(20,cxt.fc.xft)
    dR2[i,:] = -cxt.fc.dR2_mean*np.sign(cxt.fc.coeff_cv[:-1])
#%% plot for lab meeting
savedir = r'Y:\Presentations\2025\04_LabMeeting\FB6H'
i = 2
plt.close('all')
datadir = datadirs[i]
cxt = CX_tan(datadir)
cxt.reinit_fc()
taus = np.array([[ACV_taus[i,0],0.01],[ACV_taus[i,1],0.01],[0.3,0.01],[0.3,0.01],[0.3,0.01]])
cxt.fc.run(['odour onset','first odour'],partition=False,cirftau=taus)
regm,regmp =cxt.fc.set_up_regressors(['odour onset','first odour','angular velocity abs','translational vel','stationary'],
                                     cirftau=taus)
#cxt.fc.plot_example_flur()
plt.figure()
tt = cxt.fc.ts
ttp = cxt.fc.ts_y
y = cxt.fc.y
yp = cxt.fc.predy
plt.plot(ttp,y,color='k')
plt.plot(ttp,yp,color=[0.3,0.4,0.9])
plt.plot(tt,-1.5+cxt.ft2['instrip'],color='r')
plt.plot(tt,-2.5+regmp[:,2]/np.max(regmp[:,2]),color=[0.3,0.3,0.3])
plt.plot(tt,-3.5+regmp[:,3]/np.max(regmp[:,3]),color=[0.7,0.7,0.7])
plt.xlabel('time (s)')
plt.ylabel('dF/F0')
plt.savefig(os.path.join(savedir,'ACV_pulses_simple' + cxt.name+ '.pdf'))


plt.figure()
#cxt.fc.run(['first odour','translational vel'],partition=False,cirftau=taus[1:,:])
cxt.fc.run(['odour onset','first odour','angular velocity abs','translational vel'],partition=False,cirftau=taus)
tt = cxt.fc.ts
ttp = cxt.fc.ts_y
y = cxt.fc.y
yp = cxt.fc.predy
plt.plot(ttp,y,color='k')
plt.plot(ttp,yp,color=[0.3,0.4,0.9])
plt.plot(tt,-1.5+cxt.ft2['instrip'],color='r')
plt.plot(tt,-2.5+regmp[:,2]/np.max(regmp[:,2]),color=[0.3,0.3,0.3])
plt.plot(tt,-3.5+regmp[:,3]/np.max(regmp[:,3]),color=[0.7,0.7,0.7])
cxt.fc.run_dR2(20,cxt.fc.xft)
plt.xlabel('time (s)')
plt.ylabel('dF/F0')
plt.savefig(os.path.join(savedir,'ACV_pulses_motor' + cxt.name+ '.pdf'))
#%% Octanol
plt.close('all')

datadirs =[ r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial5",
           r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial4",
           r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250328\f2\Trial6",
           ]
oct_taus = np.zeros((len(datadirs),2))
dR2_oct = np.zeros((len(datadirs),3))
for i,datadir in enumerate(datadirs):
    cxt = CX_tan(datadir)
    short_tau,long_tau = tau_fitting(['oct onset','first oct'])
    taus = np.array([[short_tau,0.01],[long_tau,0.01],[0.3,0.01],[0.3,0.01],[0.3,0.01]]) # Fit first odour first
    #cxt.fc.run(['oct onset','first oct','angular velocity abs','translational vel','stationary'],partition=False,cirftau=taus)
    cxt.fc.run(['oct onset','first oct','angular velocity abs'],partition=False,cirftau=taus)
    cxt.fc.plot_example_flur()
    plt.plot(cxt.fc.ts,cxt.ft2['mfc3_stpt']/np.max(cxt.ft2['mfc3_stpt']),color='g')
    plt.title('R2 '+str(cxt.fc.r2)+' Tau 1: ' +str(short_tau) + ' Tau 2: ' +str(long_tau))
    oct_taus[i,0] = short_tau
    oct_taus[i,1] = long_tau
    cxt.fc.run_dR2(20,cxt.fc.xft)
    dR2_oct[i,:] = -cxt.fc.dR2_mean*np.sign(cxt.fc.coeff_cv[:-1])
#%%
savedir = r'Y:\Presentations\2025\04_LabMeeting\FB6H'
i = 2
plt.close('all')
datadir = datadirs[i]
cxt = CX_tan(datadir)
in_oct = cxt.ft2['mfc3_stpt']/np.max(cxt.ft2['mfc3_stpt'])
cxt.reinit_fc()
taus = np.array([[oct_taus[i,0],0.01],[oct_taus[i,1],0.01],[0.3,0.01],[0.3,0.01],[0.3,0.01]])
cxt.fc.run(['oct onset','first oct'],partition=False,cirftau=taus)
regm,regmp =cxt.fc.set_up_regressors(['odour onset','first odour','angular velocity abs','translational vel','stationary'],
                                     cirftau=taus)
#cxt.fc.plot_example_flur()
plt.figure()
tt = cxt.fc.ts
ttfull = cxt.fc.pv2['relative_time'].to_numpy()
ttp = cxt.fc.ts_y
y = cxt.fc.y
yp = cxt.fc.predy
plt.plot(ttp,y,color='k')
plt.plot(ttp,yp,color=[0.4,0.9,0.4])
plt.plot(tt,-1.5+in_oct,color=[0.4,1,0.4])
plt.plot(tt,-2.5+regmp[:,2]/np.max(regmp[:,2]),color=[0.3,0.3,0.3])
plt.plot(tt,-3.5+regmp[:,3]/np.max(regmp[:,3]),color=[0.7,0.7,0.7])
plt.xlabel('time (s)')
plt.ylabel('dF/F0')
plt.savefig(os.path.join(savedir,'Oct_pulses_simple' + cxt.name+ '.pdf'))


plt.figure()
#cxt.fc.run(['first odour','translational vel'],partition=False,cirftau=taus[1:,:])
cxt.fc.run(['oct onset','first oct','angular velocity abs','translational vel'],partition=False,cirftau=taus)
tt = cxt.fc.ts_y
y = cxt.fc.y
yp = cxt.fc.predy
plt.plot(ttp,y,color='k')
plt.plot(ttp,yp,color=[0.4,0.9,0.4])
plt.plot(ttfull,-1.5+in_oct,color=[0.4,0.8,0.4])
plt.plot(ttfull,-2.5+regmp[:,2]/np.max(regmp[:,2]),color=[0.3,0.3,0.3])
plt.plot(ttfull,-3.5+regmp[:,3]/np.max(regmp[:,3]),color=[0.7,0.7,0.7])
cxt.fc.run_dR2(20,cxt.fc.xft)
plt.xlabel('time (s)')
plt.ylabel('dF/F0')
plt.savefig(os.path.join(savedir,'Oct_pulses_motor' + cxt.name+ '.pdf'))
#%%
plt.scatter(ACV_taus[:,1],oct_taus[:,1],c='k')
plt.xlim([0,1000])
plt.ylim([0,1000])
plt.plot([0,1000],[0,1000],color='k',linestyle='--')
plt.xlabel('ACV long tau (s)',color='r',fontsize=12)
plt.ylabel('Oct long tau (s)',color=[0.4,1,0.4],fontsize=12)
plt.savefig(os.path.join(savedir,'ACV_vs_Oct_Tau' + cxt.name+ '.pdf'))
plt.savefig(os.path.join(savedir,'ACV_vs_Oct_Tau' + cxt.name+ '.png'))
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

#%% Trajectory data
# datadirs =[ r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial5",
#            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial4",
#            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250328\f2\Trial6",
#            ]
datadirs = [
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial1",#Good behaviour, strong sustained suppression during edge tracking
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250318\f1\Trial2",# Many plume entries and exits with plume disappearance

            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial1",#Good tracker
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250321\f1\Trial2",#Decent tracker multiple plumes and strong ramping
           
            
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250328\f2\Trial1", # One jump, not great tracker
            r"Y:\Data\FCI\Hedwig\FB6H_SS95649\250328\f2\Trial3",# Made a few jumps, not a great tracker 
            ]
plt.close('all')
for datadir in datadirs:
    cxt = CX_tan(datadir)
    cxt.fc.example_trajectory_jump(cxt.ca_no_nan.copy(),cxt.ft,cmin=0,cmax =1,jsize=5,cmap='magma') 
    #cxt.fc.example_trajectory_scatter(cxt.ca_no_nan.copy(),cmin=-0.5,cmax=0.5)
    plt.title(cxt.name)
    plt.ylim([-200,7000])
    plt.savefig(os.path.join(savedir,'TestTraj'+cxt.name+'.pdf'))

    cxt.fc.example_trajectory_jump(cxt.ca_rebase.copy(),cxt.ft,cmin=-0.5,cmax =0.5,jsize=5) 
    #cxt.fc.example_trajectory_scatter(cxt.ca_rebase.copy(),cmin=-0.5,cmax=0.5)
    plt.title(cxt.name)

    ft2 = cxt.ft2
    plt.figure()
    plt.plot(ft2['instrip'],color='r')
    plt.plot(np.abs(ft2['net_motion'].to_numpy()*10),color='k')
    plt.plot(cxt.ca,color=[0.2,0.2,1])
    plt.title(cxt.name)
#%% Plots for lab meeting
plt.close('all')
datadir = datadirs[0]
cxt = CX_tan(datadir)
datadir = datadirs[1]
cxt2 = CX_tan(datadir)
#%%
savedir = r'Y:\Presentations\2025\04_LabMeeting\FB6H'

ft2 = cxt.ft2
tt = cxt.fc.pv2['relative_time'].to_numpy()
plt.figure()
plt.plot(tt,ft2['instrip'],color='r')

#plt.plot(tt,-2+np.abs(ft2['net_motion'].to_numpy()*5),color='k')
ca = cxt.ca
plt.plot(tt,ca,color=[0.2,0.2,1])


ft2 = cxt2.ft2
ca = cxt2.ca
tt = cxt2.fc.pv2['relative_time'].to_numpy()

plt.plot(tt,-1.5+ft2['instrip'],color='r')

plt.plot(tt,-1.5+ca,color=[0.2,0.2,0.5])

plt.xlabel('time (s)')
plt.ylabel('dF/F0')
plt.savefig(os.path.join(savedir,'ET_SignalComparison.pdf'))
#%% Plot trajectories
plt.close('all')
cxt.fc.example_trajectory_jump(cxt.ca_no_nan.copy(),cxt.ft,cmin=-0,cmax =1,jsize=5,cmap='magma') 
plt.ylim([-200, 7000])
plt.savefig(os.path.join(savedir,'Traj' +cxt.name+'.pdf'))

cxt2.fc.example_trajectory_jump(cxt2.ca_no_nan.copy(),cxt2.ft,cmin=-0,cmax =1,jsize=5,cmap='magma') 
plt.ylim([-200, 7000])
plt.savefig(os.path.join(savedir,'Traj' +cxt2.name+'.pdf'))
#%% Plot sections of data
plt.close('all')
savedir = r'Y:\Presentations\2025\04_LabMeeting\FB6H'
section1 = [100,350]
section2 = [575,875]
section3 = [400,750]
# cxt2.fc.example_trajectory_jump(cxt2.ca_no_nan.copy(),cxt2.ft,
#                                 cmin=-0,cmax =1,jsize=5,cmap='magma',selection= section1) 
# plt.ylim([0,4500])
# # May need to deal with below manually ;()
# cxt2.fc.example_trajectory_jump(cxt2.ca_no_nan.copy(),cxt2.ft,
#                                 cmin=-0,cmax =1,jsize=5,cmap='magma',selection= section2) 
# plt.ylim([0,4500])

x,y,tt = cxt2.fc.example_trajectory_jump(cxt2.ca_no_nan.copy(),cxt2.ft,
                                cmin=-0,cmax =1,jsize=5,cmap='magma',selection= section3) 

plt.scatter(x[ug.find_nearest(tt,595.6)],y[ug.find_nearest(tt,595.6)],marker='*',color=[0.3,0.8,0.3],s=50,zorder=10)

plt.scatter(x[ug.find_nearest(tt,617.5)],y[ug.find_nearest(tt,617.5)],marker='*',color=[0.3,0.8,0.3],s=50,zorder=10)
plt.scatter(x[ug.find_nearest(tt,654.8)],y[ug.find_nearest(tt,654.8)],marker='*',color=[0.3,0.8,0.3],s=50,zorder=10)


plt.ylim([0,4500])
plt.savefig(os.path.join(savedir,'Trajectory_'+cxt2.name+'_'+str(section3[0])+'_'+str(section3[1])+'.pdf'))

ft2 = cxt2.ft2
ca = cxt2.ca
tt = cxt2.fc.pv2['relative_time'].to_numpy()
plt.figure()
plt.plot(tt,ft2['instrip'],color='r')
plt.plot(tt,ca,color=[0.2,0.2,0.5])
plt.xlim(section3)
plt.xlabel('time (s)')
plt.ylabel('dF/F0')
plt.scatter([595.6,617.5,654.8],[1.05,1.05,1.05],marker='*',color=[0.3,0.8,0.3],s=100,zorder=10)
plt.ylim([-0.05,1.1])
plt.savefig(os.path.join(savedir,'Timeseries_'+cxt2.name+'_'+str(section3[0])+'_'+str(section3[1])+'.pdf'))
#%%
plt.close('all')
plt.figure()
plt.imshow(np.zeros((10,2)),cmap='magma',vmin=0,vmax=1)
cbar = plt.colorbar(cmap='magma')
cbar.set_ticks([0,0.5,1])
plt.savefig(os.path.join(savedir,'ColourbarMagma.pdf'))
#%% Fit tau for each odour encounter
plt.close('all')

#%%
iters = 10
trials =2
out_taus = np.zeros((iters,trials))
for d in range(trials):
    datadir = datadirs[d]
    cxt = CX_tan(datadir)
    cxt.reinit_fc()
    
    ins = cxt.ft2['instrip'].to_numpy()
    insd = np.diff(ins)
    in_num = np.sum(insd>0)
    decay_TCs = [0.3,1,2,4,8,16,32,64,100,128,200,256]
    taus = np.zeros([in_num,2])
    taus[:,1] = 0.01
    
    regchoice = ['each odour sparse']
    for i in range(iters):
        print(i)
        t_tau = taus[:i+1,:]
        t_r2 = np.zeros(len(decay_TCs))
        # if i==0:
        #     t_tau = t_tau.flatten()
        for it,t in enumerate(decay_TCs):
            
            # if i==0:
            #     t_tau[i] = t
            # else:
            #     t_tau[i,0] = t 
            
            t_tau[i,0] = t 
            cxt.fc.run(regchoice,cirftau=t_tau,odour_num=i+1)
            #cxt.fc.set_up_regressors(regchoice,cirftau=t_tau,odour_num=i+1)
            t_r2[it] = cxt.fc.r2
        am = np.argmax(t_r2)
        taus[i,0] = decay_TCs[am]
        
    cxt.fc.run(regchoice,cirftau=taus[:i+1,:],odour_num=i+1)
    rm1,rm2 = cxt.fc.set_up_regressors(regchoice,cirftau=taus[:i+1,:],odour_num=i+1)
    out_taus[:,d] = taus[:i+1,0]
    #cxt.fc.run(regchoice,cirftau=[100,0.01],odour_num=1)
    cxt.fc.plot_example_flur()
    plt.title(cxt.name+ ' R2:' + str(cxt.fc.r2))
#%%
cxt.fc.plot_example_flur()
#%% Mean fluor in odour vs ISI length
import sklearn.linear_model as lm

import statsmodels.api as sm


plt.close('all')
minsize = 5
afterwinsize = 10
flynumbers = [1,1,2,2,3,3]
slopes =np.zeros((len(datadirs),2))
r2s = np.zeros((len(datadirs),2))
for i2, datadir in enumerate(datadirs):
    
    # Load and gather data
    cxt = CX_tan(datadir)
    ins = cxt.ft2['instrip'].to_numpy()
    bdx,bs = ug.find_blocks(ins>0)
    bdx = bdx[bs>=minsize]
    bs = bs[bs>=minsize]
    plotmat = np.zeros((len(bdx)-1,2))
    plotmat2 = np.zeros((len(bdx)-1,2))*np.nan
    ca = cxt.ca
    for i, b in enumerate(bdx[:-1]):
        cdx = np.arange(b,b+bs[i])
        plotmat[i,0] = np.nanmedian(ca[cdx])
        plotmat[i,1] = (bdx[i+1]-cdx[-1])*0.1#Assuming 10 Hz timebase here
        if plotmat[i,1]>afterwinsize:
            cdx2 = np.arange(b+bs[i]+(1-afterwinsize)*10,b+bs[i]+afterwinsize*10)
            plotmat2[i,0]= np.nanmedian(ca[cdx2])
            plotmat2[i,1] = (bdx[i+1]-cdx[-1])*0.1
        
        
    # Plot data and do linear fits
    plt.figure()
    plt.subplot(2,1,2)
    x = plotmat[1:,0]
    y = plotmat[1:,1]
    x = sm.add_constant(x,prepend=False)
    reg1 = sm.OLS(y,x)
    res = reg1.fit()
    r2 = np.round(res.rsquared,decimals=2)
    p = ug.round_to_sig_figs(res.pvalues[0],2)
    xp  =np.ones((2,2))
    xp[0,:] = np.min(x,axis=0)
    xp[1,:] = np.max(x,axis=0)
    yp = res.predict(xp)
    plt.scatter(x[:,0],y,color='k')
    plt.plot(xp[:,0],yp,color='k')
    plt.text(0.5,50,'R2: '+str(r2) +' p: '+str(p))
    r2s[i2,0] = r2
    slopes[i2,0] = res.params[0]
    x = plotmat2[1:,0]
    y = plotmat2[1:,1]
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    x = sm.add_constant(x,prepend=False)
    reg1 = sm.OLS(y,x)
    res = reg1.fit()
    r2 = np.round(res.rsquared,decimals=2)
    p = ug.round_to_sig_figs(res.pvalues[0],2)
    xp  =np.ones((2,2))
    xp[0,:] = np.min(x,axis=0)
    xp[1,:] = np.max(x,axis=0)
    yp = res.predict(xp)
    plt.scatter(x[:,0],y,color='r')
    plt.plot(xp[:,0],yp,color='r')
    plt.text(0.5,70,'R2: '+str(r2) +' p: '+str(p),color='r')

    plt.yscale('log')
    plt.ylabel('ISI')
    plt.xlabel('median dF/F0')
    slopes[i2,1] = res.params[0]
    r2s[i2,1] = r2
    #plt.scatter(plotmat2[:,0]-plotmat[:,0],plotmat2[:,1],color='g')
    plt.subplot(2,1,1)
    
    ft2 = cxt.ft2
    tt = cxt.fc.pv2['relative_time'].to_numpy()
    plt.plot(tt,ca,color=[0.2,0.2,1])
    plt.plot(tt,ft2['instrip'],color='r')
    ca = cxt.ca
    
    
    plt.show()
    plt.savefig(os.path.join(savedir,'Scatter_Median'+cxt.name+'.pdf'))
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(bdx[:-1],plotmat[:,0],plotmat[:,1])
#%%
plt.close('all')
sc = plt.scatter(slopes[:,0],slopes[:,1],c=flynumbers,cmap='Paired',s=np.mean(r2s,axis=1)*1000)
plt.scatter(200,200,s=500)
plt.scatter(250,250,s=250)
plt.xlabel('in plume slope')
plt.ylabel('slope 10 s post')
handles, _ = sc.legend_elements() 
plt.legend(handles, [f"Fly {num}" for num in flynumbers])
plt.savefig(os.path.join(savedir,'Slopes_R2s.pdf'))

