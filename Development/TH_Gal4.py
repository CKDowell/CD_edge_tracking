# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:51:18 2024

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
import numpy as np
#%% Imaging test for PFL3 neurons


datadir =os.path.join("Y:\Data\FCI\Hedwig\\TH_GC7f\\240306\\f1\\Trial2")
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
#%% Registration
ex = im.fly(name, datadir)
ex.register_all_images(overwrite=True)
ex.z_projection()
#%% Masks for ROI drawing
ex.mask_slice = {'All': [1,2,3,4]}
ex.t_projection_mask_slice()
#%% 
cx = CX(name,['fsb_layer'],datadir)
# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()
pv2, ft, ft2, ix = cx.load_postprocessing()
#%% 
y = pv2[['0_fsb_layer','1_fsb_layer','2_fsb_layer']]
plt.plot(y)
plt.plot(ft2['instrip'],color='k')
#%%
y = pv2['2_fsb_layer'].to_numpy()
fc = fci_regmodel(y,ft2,pv2)
fc.rebaseline(span=300)
fc.run(regchoice,partition='pre_air')
plt.plot(fc.coeffs_part[:-1],color='k')
plt.plot(fc.coeff_cv[:-1],color='r')

#%%
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
y2 = y['0_fsb_layer'].to_numpy().flatten()


yf = lowess(y2,np.arange(0,len(y)),frac=0.20)
plt.plot(y2)
plt.plot(y2-yf[:,1])
#%%
savedir = 'Y:\Data\FCI\Hedwig\TH_GC7f\SummaryFigures'
plt.close('all')
fc = fci_regmodel(pv2['0_fsb_layer'],ft2,pv2)
fc.plot_mean_flur('odour_onset')
plt.title('DA lower FSB')
plt.savefig(os.path.join(savedir,cx.name + 'DA_lower.png'))
fc = fci_regmodel(pv2['1_fsb_layer'],ft2,pv2)
fc.plot_mean_flur('odour_onset')
plt.title('DA middle FSB')
plt.savefig(os.path.join(savedir,cx.name + 'DA_middle.png'))
fc = fci_regmodel(pv2['2_fsb_layer'],ft2,pv2)
fc.plot_mean_flur('odour_onset')
plt.title('DA upper FSB')
plt.savefig(os.path.join(savedir,cx.name + 'DA_upper.png'))
#%%
regchoice = ['odour onset', 'odour offset', 'in odour', 
                            'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                
                        'angular velocity pos','x pos','x neg','y pos', 'y neg','ramp down since exit','ramp to entry']


datadirs = ["Y:\Data\FCI\Hedwig\\TH_GC7f\\240306\\f1\\Trial2",
            "Y:\Data\FCI\Hedwig\\TH_GC7f\\240314\\f2\\Trial2"]
dR2_w = np.zeros((3,len(regchoice),len(datadirs)),dtype = float)
coeffs = np.zeros_like(dR2_w)
r2 = np.zeros((3,len(datadirs)))

for ix, d in enumerate(datadirs):
    cx = CX(name,['fsb_layer'],d)
    pv2, ft, ft2, i1 = cx.load_postprocessing()
   
    for i in range(3): 
        fc = fci_regmodel(pv2[str(i) + '_fsb_layer'].to_numpy().flatten(),ft2,pv2)
        fc.rebaseline(span=500)# sorts out drift of data
        fc.run(regchoice)
        r2[i,ix] = fc.r2
        fc.run_dR2(20,fc.xft)
        dR2_w[i,:,ix] =  fc.dR2_mean
        coeffs[i,:,ix] = fc.coeff_cv[:-1]
        mn,t = fc.plot_mean_flur('odour_onset',taf=10,output=True,plotting=False)
        if i==0:
            mean_trace = np.zeros((len(mn),3))
        mean_trace[:,i] = mn
    if ix==0:
        trace_dict = {'mn_trace_' +str(ix) :  mean_trace,
                  'ts_'+str(ix): t}
    else :
        trace_dict.update({'mn_trace_' +str(ix) :  mean_trace,
                           'ts_'+str(ix): t})
#%%
fc.plot_example_flur()
#%%
plt.close('all')
colours =np.array([ [0,0,0.8], [0.4,0,0.4],[0.5,0.8,0.8]]
          )
savepath = "Y:\Presentations\\2024\\March"
plt.figure()
for ix in range(len(datadirs)):
    for i in range(3):
        plt.plot(np.linspace(0,len(regchoice)-1,len(regchoice)),dR2_w[i,:,ix],color=colours[i,:])
plt.xticks(np.linspace(0,len(regchoice)-1,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.legend(['Lower', 'Middle','Upper'])
plt.xlabel('Regressors')
plt.ylabel('delta R2')
plt.savefig(os.path.join(savepath,'TH_dR2.png'))

plt.figure()
for ix in range(len(datadirs)):
    for i in range(3):
        plt.plot(np.linspace(0,len(regchoice)-1,len(regchoice)),coeffs[i,:,ix],color=colours[i,:])
plt.xticks(np.linspace(0,len(regchoice)-1,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.legend(['Lower', 'Middle','Upper'])
plt.xlabel('Regressors')
plt.ylabel('Coefficient weights')
plt.savefig(os.path.join(savepath,'TH_Coeffs.png'))

plt.figure()
for ix in range(len(datadirs)):
    mean_trace = trace_dict['mn_trace_'+str(ix)]
    t = trace_dict['ts_'+str(ix)]
    for i in range(3):
        plt.plot(t,mean_trace[:,i],color=colours[i,:])

plt.plot([0,0],[-0.5,0.5],color='k',linestyle='--')
plt.xlabel('Time from odour onset (s)')
plt.ylabel('dF/F')