# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:16:34 2024

@author: dowel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
import os
from src.utilities import funcs as fn
import scipy.signal as signal
from src.utilities import imaging as im
#%% List of experiments where there was EB and FSB. Looks like for many exps
# Andy only imaged the FSB
#datadir = "Y:\Data\FCI\AndyData\hdb\\20220406_Fly2_001"
datadir = 'Y:\Data\FCI\AndyData\hdb\\20220520_hdb_60D05_sytjGCaMP7f_Fly1-001'
name = d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
ex = im.fly(name, datadir)
ex.z_projection()
ex.mask_slice = {'All': [1,2,3,4,5]}
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
# %%
datadir = "Y:\Data\FCI\AndyData\hdb\\20220406_Fly2_001"
name = d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
#cx = CX(name,['fsb','eb'],datadir)

#pv2, ft, ft2, ix = cx.load_postprocessing()
cxa = CX_a(datadir,Andy='hDeltaB')
plt.close('all')
cxa.simple_raw_plot()
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_phase'],cxa.amp,a_sep=5)

#%% Phase velocity
xp = fn.unwrap(cxa.phase)
xp = signal.savgol_filter(xp,30,5)
fsb_vel = np.diff(xp)
xp = fn.unwrap(cxa.phase_eb)
xp = signal.savgol_filter(xp,30,5)

eb_vel = np.diff(xp)
plt.plot(fsb_vel)
plt.plot(eb_vel)
plt.plot(cxa.ft2['instrip'])
#%%

amp = cxa.amp
amp_eb = cxa.amp_eb
phase = cxa.pdat['offset_fsb_phase']
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
line2, = ax.plot([],[],color=[1,0.2,0.2])
line3, = ax.plot([],[],lw=2,color=[0.2,0.2,1])
line, = ax.plot([], [], lw=2,color=[0.2,0.2,0.2])  # Empty line plot with line width specified
sc = ax.scatter([],[],color=[0.5,0.5,0.5])

ax.set_xticks([])
ax.set_yticks([])
x_plm = [-5, -5, 5,5]
y_plm = [0, np.max(y), np.max(y), 0]
plt.fill(x_plm,y_plm,color =[0.8,0.8,0.8])

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
savedir = "Y:\\Presentations\\2024\\MarchLabMeeting"
path_to_convert = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe'
path_to_magick = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe'
anim.save(os.path.join(savedir,'hDeltaB_eg_phase.avi'), writer=writer)

plt.show()