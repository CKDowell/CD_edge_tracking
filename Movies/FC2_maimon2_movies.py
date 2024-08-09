# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:42:31 2024

@author: dowel
"""

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
from src.utilities import funcs as fn
plt.rcParams['pdf.fonttype'] = 42 
#%%
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2"]

#%% Load data
td = 3
datadir = datadirs[td]
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
#%% Movie code

amp = cxa.amp[:,0]
amp_eb = cxa.amp_eb
web = cxa.pdat['fit_wedges_eb']
wfsb = cxa.pdat['fit_wedges_fsb_upper']
vmax_eb=np.nanpercentile(web[:],90)
vmin_eb=np.nanpercentile(web[:],25)
vmax_fsb=np.nanpercentile(wfsb[:],90)
vmin_fsb=np.nanpercentile(wfsb[:],25)

phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
phase_eb = cxa.pdat['offset_eb_phase'].to_numpy() 
import matplotlib as mpl
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg'
import networkx as nx

#mpl.use("TkAgg") 
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import ImageMagickFileWriter, ImageMagickWriter,FFMpegWriter
# Your specific x and y values
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
jumps = cxa.ft2['jump'].to_numpy()
instrip = cxa.ft2['instrip'].to_numpy()
stripdiff = np.diff(instrip)
stripon = np.where(instrip>0)[0][0]
xs = np.where(instrip==1)[0][0]
strts = np.where(stripdiff>0)[0]
stps = np.where(stripdiff<0)[0]
x = x-x[xs]
y = y-y[xs]
# Create initial line plot


fig, axs = plt.subplots(figsize=(10,10))#,ncols=3,width_ratios=[0.4,0.3,0.3])
axs.set_xticks([])
axs.set_yticks([])
axs.get_yaxis().set_visible(False)
axs.get_xaxis().set_visible(False)
# fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
# ax = axs[0]
# ax2 = axs[1]
#fig.subplots_adjust(hspace=0.2, wspace=0.2)
ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (1, 1))
# Merge the two columns in the first row

line2, = ax.plot([],[],color=[0.2,0.2,1])
line3, = ax.plot([],[],lw=2,color=[0.2,0.2,0.2])
line, = ax.plot([], [], lw=2,color=[0.2,0.2,0.2])  # Empty line plot with line width specified
px = np.array([-10,-10,10,10])
py = np.array([-10,10,10,-10])
line_e, = ax.plot([],[],lw=3,color='k',linestyle='--')
line_e2, = ax.plot([],[],lw=3,color='k',linestyle='--')

sc = ax.scatter([],[],color=[0.5,0.5,0.5])

ax.set_xticks([])
ax.set_yticks([])
ax2.set_xticks([-0.5,3.5,7.5,11.5,15.5],labels=[-180,-90,0,90,180])
ax2.set_yticks([])
ax3.set_xticks([-0.5,3.5,7.5,11.5,15.5],labels=[-180,-90,0,90,180])
ax3.set_yticks([])
ax2.set_title('EPG')
ax3.set_title('FC2')
ax2.imshow(web[:,:],cmap='gray_r',vmax=vmax_eb,vmin=vmin_eb,interpolation='none',aspect='auto')
ax3.imshow(wfsb[:,:],cmap='Blues',vmax=vmax_fsb,vmin=vmin_fsb,interpolation='none',aspect='auto')
xs = np.array([-1,16,16,-1,-1])
for i,s in enumerate(strts):
    ys = np.array([s,s,stps[i],stps[i],s])
    ax2.plot(xs,ys,color='r')
    ax3.plot(xs,ys,color='r')
    



xtarray = np.array([[-0.5,3.5,7.5,11.5,15.5],[-0.5,3.5,7.5,11.5,15.5]])
ytarray = np.zeros_like(xtarray)
ytarray[0,:] = len(wfsb)
ax2.plot(xtarray,ytarray,color='k',linestyle='--')
ax3.plot(xtarray,ytarray,color='k',linestyle='--')
ax2.set_xlim([-1.5, 16.5])
ax3.set_xlim([-1.5, 16.5])

x_plm = np.array([-5, -5, 5,5])
y_plm = [0, np.max(y), np.max(y), 0]
#ax.fill(x_plm,y_plm,color =[0.8,0.8,0.8])
#ax.fill(x_plm-210,y_plm,color =[0.8,0.8,0.8])
#ax.fill(x_plm+210,y_plm,color =[0.8,0.8,0.8])
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
    if frame>stripon:
        yf = np.array([y[frame]-10, y[frame]+10])
        xf = np.array([5,5])+jumps[frame]
        xf2 = np.array([-5,-5])+jumps[frame]
        line_e.set_data(xf,yf)
        line_e2.set_data(xf2,yf)
    ax.set_xlim(x[frame]-10,x[frame]+10)
    ax.set_ylim(y[frame]-10, y[frame]+10)
    ax2.set_ylim([frame-100,frame])
    ax3.set_ylim([frame-100,frame])
# Create animation
anim = mpl.animation.FuncAnimation(fig, update, frames=np.arange(stripon-200,len(x)), interval=10)
plt.show()
writer = FFMpegWriter(fps=20)
savedir = "Y:\Data\FCI\FCI_summaries\FC2_maimon2"
path_to_convert = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe'
path_to_magick = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe'
anim.save(os.path.join(savedir,'FC2_eg_phase' + name+'.avi'), writer=writer)

