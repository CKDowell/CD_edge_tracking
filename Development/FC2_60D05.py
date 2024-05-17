# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 18:45:50 2024

@author: dowel
"""

#%%
from analysis_funs.regression import fci_regmodel

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
#%% Imaging test for PFL3 neurons


datadir =os.path.join("Y:\Data\FCI\Hedwig\FC2_60D05_sytGC7f\\240308\\f1\\Trial4")
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
#%% Registration
ex = im.fly(name, datadir)
ex.register_all_images(overwrite=True)
ex.z_projection()
#%%
ex.mask_slice = {'All': [1,2,3,4]}
ex.t_projection_mask_slice()
#%%

cx = CX(name,['fsb','eb'],datadir)
# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()
pv2, ft, ft2, ix = cx.load_postprocessing()

#%%
cxa = CX_a(datadir)
#%%
plt.close('all')
cxa.simple_raw_plot()
plt.rcParams['pdf.fonttype'] = 42 
plt.savefig(os.path.join(datadir ,'Simple_RawPhase' +'.pdf'), format='pdf')
plt.savefig(os.path.join(datadir ,'Simple_RawPhase' +'.png'))
cxa.mean_in_plume()
plt.savefig(os.path.join(datadir,'In_plume_phase' +'.png'))
cxa.mean_phase_trans()
plt.savefig(os.path.join(datadir, 'PlumeEntry_phase' +'.png'))
cxa.mean_phase_arrow()
plt.savefig(os.path.join(datadir, 'MeanPhase_arrow' +'.png'))
plt.savefig(os.path.join(datadir, 'MeanPhase_arrow' +'.pdf'))
#%%
phase,phase_offset,amp = cx.unyoked_phase('fsb')
phase_eb,phase_offset_eb,amp_eb = cx.unyoked_phase('eb')
pdat = cx.phase_yoke('eb',['fsb'])
#%%
pon = ft2['instrip'].to_numpy(dtype=int)
x =180* pdat['offset_eb_phase']/np.pi
y = 180*pdat['offset_fsb_phase']/np.pi
vel = np.sqrt(ft2['x_velocity'].to_numpy()**2+ft2['y_velocity'].to_numpy()**2)
plt.scatter(x[vel<1],y[vel<1],color='k',s=2)
plt.scatter(x[vel>1],y[vel>1],color='r',s=5)

plt.scatter(x[pon==0],y[pon==0],color='k',s=2)
plt.scatter(x[pon==1],y[pon==1],color='r',s=5)
plt.xlabel('Elipsoid body phase')
plt.ylabel('FSB phase')
#%%

plt.close('all')
ebs = []
for i in range(16):
    ebs.append(str(i) +'_eb')
for i in range(16):
    ebs.append(str(i) +'_fsb')
plt.figure()
eb = pv2[ebs]
t = np.arange(0,len(eb))
plt.imshow(eb, interpolation='None',aspect='auto')
new_phase = np.interp(phase_eb, (phase_eb.min(), phase_eb.max()), (-0.5, 15.5))
#plt.plot(new_phase,t,color='r',linewidth=0.5)
plt.plot([15.5,15.5],[min(t), max(t)],color='r')


new_phase = np.interp(phase, (phase.min(), phase.max()), (15.5, 31.5))
#plt.plot(new_phase,t,color='r',linewidth=0.5)

new_heading = ft2['ft_heading'].to_numpy()
new_heading = np.interp(new_heading, (new_heading.min(), new_heading.max()), (32, 48))
plt.plot(new_heading,t,color='k')
plt.plot(ft2['instrip']*10+40,t)


plt.show()



plt.figure()

plt.plot(ft2['ft_posx'],ft2['ft_posy'],color='k')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

plt.figure()
t2 = pv2['relative_time'].to_numpy()
vel = np.sqrt(ft2['x_velocity'].to_numpy()**2+ft2['y_velocity'].to_numpy()**2)
plt.plot(t2,ft2['instrip'],color='k')
plt.plot(t2,pdat['offset_fsb_phase'],color='r')
plt.plot(t2,ft2['ft_heading'],color='k')
plt.show()
#%%
fc = fci_regmodel(phase_eb,ft2,pv2)
fc.plot_mean_flur('odour_onset')
#%%
amp = cxa.amp
amp_eb = cxa.amp_eb
phase = cxa.pdat['offset_fsb_upper_phase']
phase_eb = cxa.pdat['offset_eb_phase'] 
import matplotlib as mpl
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg'
import networkx as nx

#mpl.use("TkAgg") 
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import ImageMagickFileWriter, ImageMagickWriter,FFMpegWriter
# Your specific x and y values
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()

instrip = cxa.ft2['instrip'].to_numpy()
xs = np.where(instrip==1)[0][0]
x = x-x[xs]
y = y-y[xs]
# Create initial line plot

fig, ax = plt.subplots(figsize=(10,10))
line2, = ax.plot([],[],color=[0.2,0.2,1])
line3, = ax.plot([],[],lw=2,color=[0.2,0.2,0.2])
line, = ax.plot([], [], lw=2,color=[0.2,0.2,0.2])  # Empty line plot with line width specified
sc = ax.scatter([],[],color=[0.5,0.5,0.5])

ax.set_xticks([])
ax.set_yticks([])
x_plm = [-5, -5, 5,5]
y_plm = [0, np.max(y), np.max(y), 0]
plt.fill(x_plm,y_plm,color =[0.8,0.8,0.8])
plt.fill(x_plm-210,y_plm,color =[0.8,0.8,0.8])
plt.fill(x_plm+210,y_plm,color =[0.8,0.8,0.8])
ax.set_aspect('equal')
# Set axis limits

xa = 50*amp*np.sin(phase)+x
ya = 50*amp*np.cos(phase)+y
xa2 = 50*amp_eb*np.sin(phase_eb)+x
ya2 = 50*amp_eb*np.cos(phase_eb)+y
# Animation update function
def update(frame):
    # Update the line plot with the current x and y values
    line2.set_data([x[frame],xa[frame] ], [y[frame],ya[frame]])
    line3.set_data([x[frame],xa2[frame] ], [y[frame],ya2[frame]])
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
savedir = "Y:\Data\FCI\FCI_summaries\FC2_maimon2"
path_to_convert = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe'
path_to_magick = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe'
anim.save(os.path.join(savedir,'FC2_eg_phase.avi'), writer=writer)

plt.show()

#%% 

