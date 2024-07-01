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


for i in [1]:
    datadir =os.path.join("Y:\\Data\\FCI\\Hedwig\\TH_GC7f\\240529\\f3\\Trial"+str(i))
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
plt.close('all')
savepath = "Y:\\Presentations\\2024\\MarchLabMeeting\\Figures"
plt.rcParams['pdf.fonttype'] = 42 

for i in range(3):
    tstr = str(i) + '_fsb_layer'

    y = pv2[tstr].to_numpy()
    
    fc = fci_regmodel(y,ft2,pv2)
    fc.rebaseline(span=500)
    
    mn = np.percentile(fc.ca,2.5)
    mx = np.percentile(fc.ca,97.5)
    fc.example_trajectory(cmin=-.5,cmax=.5)
    plt.ylim([-30,200])
    plt.xlim([-20,20])
    plt.savefig(os.path.join(savepath,'TrajFlur'+ str(i) + '.png'))
    plt.savefig(os.path.join(savepath,'TrajFlur'+ str(i) + '.pdf'))
#%%
plt.imshow(r2,cmap='coolwarm',vmin=-0.5,vmax=0.5)
plt.colorbar(ticks = [-0.5,-0.25,0,0.25,0.5])

# %% 
plt.close('all')
colours = np.array([[79,0,148],[212,0,149],[255,170,239]])/255
datadirs = ["Y:\Data\FCI\Hedwig\\TH_GC7f\\240306\\f1\\Trial2",
            "Y:\Data\FCI\Hedwig\\TH_GC7f\\240314\\f2\\Trial2"]
for  d in datadirs:
    plt.figure()
    cx = CX(name,['fsb_layer'],d)
    pv2, ft, ft2, ix = cx.load_postprocessing()
    for i in range(3):
        tstr = str(i) + '_fsb_layer'
    
        y = pv2[tstr].to_numpy()
        fc = fci_regmodel(y,ft2,pv2)
        fc.rebaseline(span=500)
        plt.plot(fc.ts,fc.ca,color=colours[i,:])
    plt.plot(fc.ts,fc.ft2['instrip'],color='k')
    plt.show()
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
                        'angular velocity pos','angular velocity neg',
                        #'x pos','x neg','y pos', 'y neg',
                        'translational vel',
                        'ramp down since exit','ramp to entry']


datadirs = ["Y:\Data\FCI\Hedwig\\TH_GC7f\\240306\\f1\\Trial2",
            "Y:\Data\FCI\Hedwig\\TH_GC7f\\240314\\f2\\Trial2"]
dR2_w = np.zeros((3,len(regchoice),len(datadirs)),dtype = float)
coeffs = np.zeros_like(dR2_w)
r2 = np.zeros((3,len(datadirs)))
corr = np.zeros((3,3,len(datadirs)))
for ix, d in enumerate(datadirs):
    cx = CX(name,['fsb_layer'],d)
    pv2, ft, ft2, i1 = cx.load_postprocessing()
    y = pv2[['0_fsb_layer','1_fsb_layer','2_fsb_layer']]
    for i in range(3): 
        fc = fci_regmodel(pv2[str(i) + '_fsb_layer'].to_numpy().flatten(),ft2,pv2)
        fc.rebaseline(span=500)# sorts out drift of data
        y[str(i) + '_fsb_layer'] = fc.ca
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
    corr[:,:,ix] = y.corr()
#%%
plt.imshow(corr[:,:,1],vmin=-1,vmax=1,cmap= 'coolwarm')
#%%
plt.close('all')
colours = np.array([[79,0,148],[212,0,149],[255,170,239]])/255
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
sign_dR2 = -dR2_w*np.sign(coeffs)
for ix in range(len(datadirs)):
    for i in range(3):
        plt.plot(np.linspace(0,len(regchoice)-1,len(regchoice)),sign_dR2[i,:,ix],color=colours[i,:])
plt.xticks(np.linspace(0,len(regchoice)-1,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.legend(['Lower', 'Middle','Upper'])
plt.xlabel('Regressors')
plt.ylabel('delta R2 * sign(coeffs)')
plt.savefig(os.path.join(savepath,'TH_dR2_signed.png'))


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
#%% Movie of lower middle and upper layers

for i in range(3):
    
    
    fc = fci_regmodel(pv2[str(i) + '_fsb_layer'].to_numpy(),ft2,pv2)
    fc.rebaseline(span=500)
    if i==0:
        y = np.zeros((len(fc.ca),3))
        y[:,i] = fc.ca
    else :
        y[:,i] = fc.ca
yca = y        
#%%

import matplotlib as mpl
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg'
import networkx as nx

#mpl.use("TkAgg") 
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import ImageMagickFileWriter, ImageMagickWriter,FFMpegWriter
# Your specific x and y values
x = ft2['ft_posx'].to_numpy()
y = ft2['ft_posy'].to_numpy()
x,y = fc.fictrac_repair(x,y)
instrip = ft2['instrip'].to_numpy()
xs = np.where(instrip==1)[0][0]
x = x-x[xs]
y = y-y[xs]
# Create initial line plot

fig, ax = plt.subplots(figsize=(10,10))
line2, = ax.plot([],[],lw=8,color=colours[0,:])
line3, = ax.plot([],[],lw=8,color=colours[1,:])
line4, = ax.plot([],[],lw=8,color=colours[2,:])
line, = ax.plot([], [], lw=2,color=[0.2,0.2,0.2])  # Empty line plot with line width specified
sc = ax.scatter([],[],color=[0.5,0.5,0.5])

ax.set_xticks([])
ax.set_yticks([])
x_plm = [-5, -5, 5,5]
y_plm = [0, np.max(y), np.max(y), 0]
plt.fill(x_plm,y_plm,color =[0.8,0.8,0.8])

ax.set_aspect('equal')
# Set axis limits
mult = 4
ya = mult*yca[:,0]+y

ya2 = mult*yca[:,1]+y
ya3 = mult*yca[:,2]+y
# Animation update function
def update(frame):
    # Update the line plot with the current x and y values
    line2.set_data([x[frame]-1,x[frame]-1 ], [y[frame],ya[frame]])
    line3.set_data([x[frame],x[frame] ], [y[frame],ya2[frame]])
    line4.set_data([x[frame]+1,x[frame]+1 ], [y[frame],ya3[frame]])
    if frame>100:
        line.set_data(x[frame-100:frame], y[frame-100:frame])
    else:
        line.set_data(x[:frame], y[:frame])
    
    if instrip[frame]>0:
        sc.set_offsets(np.column_stack((x[frame],y[frame])))
        
    ax.set_xlim(x[frame]-10,x[frame]+10)
    ax.set_ylim(y[frame]-10, y[frame]+10)
    
# Create animation
anim = mpl.animation.FuncAnimation(fig, update, frames=len(x), interval=10)
writer = FFMpegWriter(fps=20)
savedir = "Y:\\Presentations\\2024\\MarchLabMeeting"
path_to_convert = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe'
path_to_magick = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe'
anim.save(os.path.join(savedir,'Da_activity.avi'), writer=writer)

plt.show()