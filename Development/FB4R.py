# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:47:27 2024

@author: dowel
"""

from analysis_funs.regression import fci_regmodel
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_tan import CX_tan
import numpy as np
from Utilities.utils_general import utils_general as ug

#%% 



for i in [3]:
    datadir =os.path.join("Y:\Data\FCI\\Hedwig\\SS61645_FB4R\\240911\\f1\\Trial"+str(i))
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
for i in [3]:
    datadir =os.path.join("Y:\Data\FCI\\Hedwig\\SS61645_FB4R\\240911\\f1\\Trial"+str(i))
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


#pv2, ft, ft2, ix = cx.load_postprocessing()
#%% Plot example data
#datadir = 
datadirs = ["Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240828\\f3\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240910\\f1\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61645_FB4R\\240911\\f1\\Trial3" # Only 2 jumps
            
            ]
plt.close('all')
for d in datadirs:
    cxt = CX_tan(d) 
    
    cxt.fc.example_trajectory_jump(cxt.fc.ca,cxt.ft,cmin=-0.4,cmax =0.4) 
    plt.figure()
    cxt.fc.mean_traj_nF_jump(cxt.fc.ca,plotjumps=True)
    
    ca = cxt.pv2['0_fsbtn'].to_numpy()
    ins = cxt.ft2['instrip'].to_numpy()
    u = ug()
    x = cxt.ft2['ft_posx'].to_numpy()
    y = cxt.ft2['ft_posy'].to_numpy()
    
    tt = cxt.pv2['relative_time'].to_numpy()
    dx,dx,dd = u.get_velocity(x,y,tt)
    plt.figure()
    plt.plot(tt,ca,color='k')
    plt.plot(tt,-.25+ins*.2,color='r')
    plt.plot(tt[1:],(dd/20)-1,color=[.5,.5,.5])
    
#%%
datadirs = ["Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240828\\f3\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240910\\f1\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61645_FB4R\\240911\\f1\\Trial3" # Only 2 jumps
            
            ]
plt.close('all')
for d in datadirs:
    cxt = CX_tan(d) 
    plt.figure()
    fb5ab = cxt.pv2['0_fsbtn'].to_numpy()
   
    #hdc2 = np.mean(cxa.pdat['wedges_fsb2_ch2'],axis=1)
    t = cxt.pv2['relative_time'].to_numpy()

    plt.plot(t,fb5ab,color='k')
    #plt.plot(t,-fb5ab+1,color='m')
    #plt.plot(t,hdc2,color='b')
    ins = cxt.ft2['instrip'].to_numpy().astype(float)

    if np.sum(ins)>0:
        #plt.plot(t,ins,color='r')
        plt.fill_between(t,ins*0-1,ins*.5-1,color=[1,.2,.2])
    else:
        ins = cxt.ft2['mfc3_stpt'].to_numpy()>0
        plt.plot(t,ins,color='g')
    #plt.plot(t,ins*2.25-1,color=[1,.2,.2])
    plt.ylabel('dF/F0')
    plt.xlabel('time (s)')
#%%




fc = fci_regmodel(cxt.pv2[['0_fsbtn']].to_numpy().flatten(),cxt.ft2,cxt.pv2)
fc.rebaseline(span=500,plotfig=True)
#%%
y = fc.ca
plt.figure()
plt.plot(y)
plt.plot(ft2['instrip'],color='k')

fc = fci_regmodel(y,ft2,pv2)
fc.example_trajectory(cmin=-0.2,cmax=0.2)
#%% Reg model
plt.close('all')

regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos',
                                #'translational vel dirs',
                                'translational vel',
                                'ramp down since exit','ramp to entry']
datadirs = ["Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240828\\f3\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240910\\f1\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61645_FB4R\\240911\\f1\\Trial3" # Only 2 jumps
            
            ]
for d in datadirs:
    cxt = CX_tan(d) 
    fc = cxt.fc
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    
    
    plt.figure(1)
    plt.plot(fc.dR2_mean)
    plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.ylabel('delta R2')
    plt.xlabel('Regressor name')
    plt.show()
    
    plt.figure(2)
    plt.plot(fc.coeff_cv[:-1])
    plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.ylabel('Coefficient weight')
    plt.xlabel('Regressor name')
    plt.show()
    
    plt.figure(3)
    plt.plot([0,len(regchoice)],[0, 0],color='k',linestyle='--') 
    plt.plot(-fc.dR2_mean*np.sign(fc.coeff_cv[:-1]),color='k')
    plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.ylabel('delta R2 * sign(coeffs)')
    fc.plot_example_flur()
    plt.xlabel('Regressor name')
    plt.show()
#%% Response by odour onset
plt.figure()
for d in datadirs:
    cxt = CX_tan(d) 
    jumps = cxt.get_entries_exits_like_jumps() 
    ca = cxt.ca.copy()
    ent_df = np.zeros(len(jumps))
    for i, j in enumerate(jumps):
        dx = np.arange(j[0],j[1])
        dx2 = np.arange(j[0]-20,j[0])
        ent_df[i] = np.median(ca[dx])-np.median(ca[dx2])
    plt.plot(ent_df)

#%% Reg model with movement
plt.close('all')

regchoice = [
                                'angular velocity pos',
                                #'translational vel dirs',
                                'translational vel',
                                ]
datadirs = ["Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240828\\f3\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240910\\f1\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61645_FB4R\\240911\\f1\\Trial3" # Only 2 jumps
            
            ]
for d in datadirs:
    cxt = CX_tan(d) 
    fc = cxt.fc
    fc.run(regchoice,partition='pre_air')
    fc.plot_example_flur(parts=True)

    
#%%
plt.close('all')

regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'translational vel dirs',
                                ]
reglabels =['odour onset', 'odour offset', 'in odour', 
                                'tv -157.5','tv -112.5', 'tv -67.5', 
                                'tv -22.5', 'tv  22.5',   'tv 67.5',  
                                'tv 112.5', 'tv 157.5'
                                ]

datadirs = ["Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240828\\f3\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240910\\f1\\Trial1",
            "Y:\Data\FCI\\Hedwig\\SS61645_FB4R\\240911\\f1\\Trial3" # Only 2 jumps
            
            ]
for d in datadirs:
    cxt = CX_tan(d) 
    fc = cxt.fc
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    
    plt.figure(1)
    plt.plot(fc.dR2_mean)
    plt.xticks(np.arange(0,len(reglabels)),labels=reglabels,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.ylabel('delta R2')
    plt.xlabel('Regressor name')
    plt.show()
    
    plt.figure(2)
    plt.plot(fc.coeff_cv[:-1])
    plt.xticks(np.arange(0,len(reglabels)),labels=reglabels,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.ylabel('Coefficient weight')
    plt.xlabel('Regressor name')
    plt.show()
    
    plt.figure(3)
    plt.plot([0,len(regchoice)],[0, 0],color='k',linestyle='--') 
    plt.plot(-fc.dR2_mean*np.sign(fc.coeff_cv[:-1]),color='k')
    plt.xticks(np.arange(0,len(reglabels)),labels=reglabels,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.ylabel('delta R2 * sign(coeffs)')
    fc.plot_example_flur()
    plt.xlabel('Regressor name')
    plt.show()