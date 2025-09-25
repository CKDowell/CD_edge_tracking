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
from scipy.stats import circmean, circstd

from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
from src.utilities import funcs as fn
from Utilities.utils_general import utils_general as ug
plt.rcParams['pdf.fonttype'] = 42 
#%%
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial5",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial2",
r"Y:\Data\FCI\Hedwig\FC2_maimon2\250128\f1\Trial4"]

#%% Load data
td =3
datadir = datadirs[td]
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
#%% Movie code version 1

amp = cxa.amp[:,0]
amp_eb = cxa.amp_eb
web = cxa.pdat['fit_wedges_eb']
wfsb = cxa.pdat['fit_wedges_fsb_upper']
vmax_eb=np.nanpercentile(web[:],90)
vmin_eb=np.nanpercentile(web[:],25)
vmax_fsb=np.nanpercentile(wfsb[:],90)
vmin_fsb=np.nanpercentile(wfsb[:],25)

phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
phase = ug.savgol_circ(phase, 30, 3)


phase_eb = cxa.pdat['offset_eb_phase'].to_numpy() 
phase_eb = ug.savgol_circ(phase_eb, 30, 3)

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
axs.axis("off")
# fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
# ax = axs[0]
# ax2 = axs[1]
#fig.subplots_adjust(hspace=0.2, wspace=0.2)
ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (1, 1))

for axis in [ax, ax2, ax3]:
    axis.spines['top'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['right'].set_visible(False)

    if axis in [ax2, ax3]:
        axis.spines['bottom'].set_visible(True)
        axis.xaxis.set_visible(True)
    else:
        axis.spines['bottom'].set_visible(False)

# Merge the two columns in the first row

line2, = ax.plot([],[],lw=3,color=[0.2,0.2,1])
line3, = ax.plot([],[],lw=3,color=[0.2,0.2,0.2])
line, = ax.plot([], [], lw=1,color=[0.2,0.2,0.2])  # Empty line plot with line width specified
px = np.array([-10,-10,10,10])
py = np.array([-10,10,10,-10])
line_e, = ax.plot([],[],lw=3,color='k',linestyle='--')
line_e2, = ax.plot([],[],lw=3,color='k',linestyle='--')

sc = ax.scatter([],[],color=[1,0.5,0.5],s=100)

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
plt.box('False')
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
#anim.save(os.path.join(savedir,'FC2_eg_phase' + name+'.avi'), writer=writer)
#%% Movie code version 2

amp = cxa.amp[:,0]
amp_eb = cxa.amp_eb
web = cxa.pdat['wedges_offset_eb'].copy()
wfsb = cxa.pdat['wedges_offset_fsb_upper'].copy()

for i in range(web.shape[1]):
    web[:,i] = sg.savgol_filter(web[:,i],30,2)
    wfsb[:,i] = sg.savgol_filter(wfsb[:,i],30,2)

vmax_eb=np.nanpercentile(web[:],90)
vmin_eb=np.nanpercentile(web[:],25)
vmax_fsb=np.nanpercentile(wfsb[:],90)
vmin_fsb=np.nanpercentile(wfsb[:],25)

phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
phase = ug.savgol_circ(phase, 30, 3)


phase_eb = cxa.pdat['offset_eb_phase'].to_numpy() 
phase_eb = ug.savgol_circ(phase_eb, 30, 3)

import matplotlib as mpl
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg'
import networkx as nx

#mpl.use("TkAgg") 
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import ImageMagickFileWriter, ImageMagickWriter,FFMpegWriter

# Your specific x and y values
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
x,y = cxa.fictrac_repair(x,y)
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


fig, axs = plt.subplots(figsize=(15,8))#,ncols=3,width_ratios=[0.4,0.3,0.3])
axs.set_xticks([])
axs.set_yticks([])
axs.get_yaxis().set_visible(False)
axs.get_xaxis().set_visible(False)
axs.axis("off")
# fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
# ax = axs[0]
# ax2 = axs[1]

#fig.subplots_adjust(hspace=0.2, wspace=0.2)

#Arrows
ax = plt.subplot2grid((1, 3), (0, 0), colspan=1)

# Trajectory
ax2 = plt.subplot2grid((1, 3), (0, 1))

# Wedges
ax3 = plt.subplot2grid((1, 3), (0, 2))

ax2.set_xticks([])
ax2.set_yticks([])
ax.set_xticks([])
ax.set_yticks([])
ax3.set_yticks([])
ax3.set_xticks([-180,-90,0,90,180])
for axis in [ax, ax2, ax3]:
    axis.spines['top'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['right'].set_visible(False)

    if axis in [ ax3]:
        axis.spines['bottom'].set_visible(True)
        axis.xaxis.set_visible(True)
    else:
        axis.spines['bottom'].set_visible(False)

line_fc, = ax.plot([],[],lw=3,color=[0.2,0.2,1])
line_eb, = ax.plot([],[],lw=3,color=[0.2,0.2,0.2])



line, = ax2.plot([],[],lw=2,color='k')

xa = np.sin(phase)
ya = np.cos(phase)
xa2 = np.sin(phase_eb)
ya2 = np.cos(phase_eb)

ax.set_xlim(-1.2,1.2)
ax.set_ylim(-1.2,1.2)
ax.set_aspect('equal')

ax2.set_aspect('equal')
xrange2 = np.linspace(-180,180,16)
ax3.set_xlim(-180,180)
ax3.set_ylim(-1.3,1.3)
ax3.set_xlabel('Column phase (deg)')
ax3.plot([-180,180],[0,0],color='k',linestyle='--')
ax3.plot([-180,180],[-1,-1],color='k',linestyle='--')
ax3.plot([0,0],[-1,-0.25],color='k',linestyle='--')
ax3.plot([0,0],[0,.75],color='k',linestyle='--')
ax3.plot([90,90],[0,.75],color='r',linestyle='--')
ax3.plot([90,90],[-1,-0.25],color='r',linestyle='--')

ft2 = cxa.ft2

ins = ft2['instrip'].to_numpy()
jumps = ft2['jump'].to_numpy()
tt = cxa.pv2['relative_time'].to_numpy()
inplume = ins>0
st  = np.where(ins)[0][0]
x = x-x[st-1]
y = y-y[st-1]
  



jumps = jumps-np.mod(jumps,3)
jd = np.diff(jumps)
jn = np.where(np.abs(jd)>0)[0]+1
print(jumps[jn])
jkeep = np.where(np.diff(jn)>1)[0]

xrange = np.max(x)-np.min(x)
yrange = np.max(y)-np.min(y)

mrange = np.max([xrange,yrange])+100
y_med = yrange/2
x_med = xrange/2
ylims = [y_med-mrange/2, y_med+mrange/2]

xlims = [x_med-mrange/2, x_med+mrange/2]
yj = y[jn]
yj = np.append(yj,y[-1])
tj = 0
x1 = 0+5+tj
x2 = 0-5+tj
y1 = 0
y2 = yj[0]
xvec = np.array([x1,x2,x2,x1])
yvec = [y1,y1,y2,y2]

cents = [-630,-420,-210, 0,210,420,630]
ax2.fill(xvec,yvec,color=[0.7,0.7,0.7])
for c in cents:
    ax2.fill(xvec+c,yvec,color=[0.7,0.7,0.7])
    
for i,j in enumerate(jn):
    
    tj = jumps[j]
    x1 = 0+5+tj
    x2 = 0-5+tj
    y1 = yj[i]
    y2 = yj[i+1]
    xvec = np.array([x1,x2,x2,x1])
    yvec = [y1,y1,y2,y2]
    for c in cents:
        ax2.fill(xvec+c,yvec,color=[0.7,0.7,0.7])
        #ax2.plot([xvec[1],xvec[1]-20],yvec[0:2],color=colours[3,:],linewidth=0.75)
        #ax2.scatter(xvec[1]-20,yvec[0],color=colours[3,:],marker='<')

hist_fc, =ax3.plot([],[],lw=3,color=[0.2,0.2,1])
hist_eb, =ax3.plot([],[],lw=3,color=[0.2,0.2,0.2])

def update(frame):
    line_fc.set_data([0,xa[frame] ], [0,ya[frame]])
    line_eb.set_data([0,xa2[frame] ], [0,ya2[frame]])
    
    hist_fc.set_data(xrange2,wfsb[frame,:])
    hist_eb.set_data(xrange2,web[frame,:]-1)
    
    if frame>100:
        line.set_data(x[frame-100:frame], y[frame-100:frame])
    else:
        line.set_data(x[:frame], y[:frame])
    ax2.set_xlim(x[frame]-10,x[frame]+10)
    ax2.set_ylim(y[frame]-10, y[frame]+10)

# Create animation
anim = mpl.animation.FuncAnimation(fig, update, frames=np.arange(stripon-200,len(x)), interval=10)
plt.show()
writer = FFMpegWriter(fps=20)
savedir = "Y:\Data\FCI\FCI_summaries\FC2_maimon2"
path_to_convert = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe'
path_to_magick = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe'
#anim.save(os.path.join(savedir,'FC2_eg_phase_pt2_' + name+'.avi'), writer=writer)

#%% Movie code version 2 _black

amp = cxa.amp[:,0]
amp_eb = cxa.amp_eb
web = cxa.pdat['wedges_offset_eb'].copy()
wfsb = cxa.pdat['wedges_offset_fsb_upper'].copy()

for i in range(web.shape[1]):
    web[:,i] = sg.savgol_filter(web[:,i],30,2)
    wfsb[:,i] = sg.savgol_filter(wfsb[:,i],30,2)

vmax_eb=np.nanpercentile(web[:],90)
vmin_eb=np.nanpercentile(web[:],25)
vmax_fsb=np.nanpercentile(wfsb[:],90)
vmin_fsb=np.nanpercentile(wfsb[:],25)

phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
phase = ug.savgol_circ(phase, 30, 3)


phase_eb = cxa.pdat['offset_eb_phase'].to_numpy() 
phase_eb = ug.savgol_circ(phase_eb, 30, 3)

import matplotlib as mpl
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg'
import networkx as nx

#mpl.use("TkAgg") 
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import ImageMagickFileWriter, ImageMagickWriter,FFMpegWriter

# Your specific x and y values
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
x,y = cxa.fictrac_repair(x,y)
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


fig, axs = plt.subplots(figsize=(15,8))#,ncols=3,width_ratios=[0.4,0.3,0.3])
fig.set_facecolor('black')
axs.set_xticks([])
axs.set_yticks([])
axs.get_yaxis().set_visible(False)
axs.get_xaxis().set_visible(False)
axs.axis("off")
# fig, axs = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
# ax = axs[0]
# ax2 = axs[1]

#fig.subplots_adjust(hspace=0.2, wspace=0.2)

#Arrows
ax = plt.subplot2grid((1, 3), (0, 0), colspan=1)
ax.set_facecolor('black')
# Trajectory
ax2 = plt.subplot2grid((1, 3), (0, 1))
ax2.set_facecolor('black')
# Wedges
ax3 = plt.subplot2grid((1, 3), (0, 2))
ax3.set_facecolor('black')

ax2.set_xticks([])
ax2.set_yticks([])
ax.set_xticks([])
ax.set_yticks([])
ax3.set_yticks([])
ax3.set_xticks([-180,-90,0,90,180])
ax3.tick_params(axis='x', colors='white')
ax3.spines['bottom'].set_color('white')
for axis in [ax, ax2, ax3]:
    axis.spines['top'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['right'].set_visible(False)

    if axis in [ ax3]:
        axis.spines['bottom'].set_visible(True)
        axis.xaxis.set_visible(True)
    else:
        axis.spines['bottom'].set_visible(False)

line_fc, = ax.plot([],[],lw=3,color=[1,0.2,0.2])
line_eb, = ax.plot([],[],lw=3,color=[1,1,1])



line, = ax2.plot([],[],lw=2,color='w')
line2, = ax2.plot([],[],lw=2,color='r')

xa = np.sin(phase)
ya = np.cos(phase)
xa2 = np.sin(phase_eb)
ya2 = np.cos(phase_eb)

ax.set_xlim(-1.2,1.2)
ax.set_ylim(-1.2,1.2)
ax.set_aspect('equal')

ax2.set_aspect('equal')
xrange2 = np.linspace(-180,180,16)
ax3.set_xlim(-180,180)
ax3.set_ylim(-1.3,1.3)
ax3.set_xlabel('Column phase (deg)')
#ax3.plot([-180,180],[0,0],color='w',linestyle='--')
#ax3.plot([-180,180],[-1,-1],color='w',linestyle='--')
# ax3.plot([0,0],[-1,-0.25],color='k',linestyle='--')
# ax3.plot([0,0],[0,.75],color='k',linestyle='--')
# ax3.plot([90,90],[0,.75],color='r',linestyle='--')
# ax3.plot([90,90],[-1,-0.25],color='r',linestyle='--')

ft2 = cxa.ft2

ins = ft2['instrip'].to_numpy()
jumps = ft2['jump'].to_numpy()
tt = cxa.pv2['relative_time'].to_numpy()
inplume = ins>0
st  = np.where(ins)[0][0]
x = x-x[st-1]
y = y-y[st-1]
  



jumps = jumps-np.mod(jumps,3)
jd = np.diff(jumps)
jn = np.where(np.abs(jd)>0)[0]+1
print(jumps[jn])
jkeep = np.where(np.diff(jn)>1)[0]

xrange = np.max(x)-np.min(x)
yrange = np.max(y)-np.min(y)

mrange = np.max([xrange,yrange])+100
y_med = yrange/2
x_med = xrange/2
ylims = [y_med-mrange/2, y_med+mrange/2]

xlims = [x_med-mrange/2, x_med+mrange/2]
yj = y[jn]
yj = np.append(yj,y[-1])
tj = 0
x1 = 0+5+tj
x2 = 0-5+tj
y1 = 0
y2 = yj[0]
xvec = np.array([x1,x2,x2,x1])
yvec = [y1,y1,y2,y2]

cents = [-630,-420,-210, 0,210,420,630]
ax2.fill(xvec,yvec,color=[0.7,0.7,0.7])
for c in cents:
    ax2.fill(xvec+c,yvec,color=[0.7,0.7,0.7])
    
for i,j in enumerate(jn):
    
    tj = jumps[j]
    x1 = 0+5+tj
    x2 = 0-5+tj
    y1 = yj[i]
    y2 = yj[i+1]
    xvec = np.array([x1,x2,x2,x1])
    yvec = [y1,y1,y2,y2]
    for c in cents:
        ax2.fill(xvec+c,yvec,color=[0.7,0.7,0.7])
        #ax2.plot([xvec[1],xvec[1]-20],yvec[0:2],color=colours[3,:],linewidth=0.75)
        #ax2.scatter(xvec[1]-20,yvec[0],color=colours[3,:],marker='<')

hist_fc, =ax3.plot([],[],lw=3,color=[1,0.2,0.2])
hist_eb, =ax3.plot([],[],lw=3,color=[1,1,1])

def update(frame):
    line_fc.set_data([0,xa[frame] ], [0,ya[frame]])
    line_eb.set_data([0,xa2[frame] ], [0,ya2[frame]])
    
    hist_fc.set_data(xrange2,wfsb[frame,:])
    hist_eb.set_data(xrange2,web[frame,:]-1)
    
    if frame>100:
        fdx = np.arange(frame-100,frame)
    else:
        fdx = np.arange(0,frame)
        
    tx = x[fdx]
    ty = y[fdx]
    tins = instrip[fdx]
    line.set_data(tx,ty)
    tx[tins==0] =np.nan
    ty[tins==0] = np.nan
    line2.set_data(tx,ty)
    ax2.set_xlim(x[frame]-10,x[frame]+10)
    ax2.set_ylim(y[frame]-10, y[frame]+10)

# Create animation
anim = mpl.animation.FuncAnimation(fig, update, frames=np.arange(stripon-200,len(x)), interval=10)
plt.show()
writer = FFMpegWriter(fps=20)
savedir = "Y:\Data\FCI\FCI_summaries\FC2_maimon2"
path_to_convert = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe'
path_to_magick = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe'
anim.save(os.path.join(savedir,'FC2_eg_phase_pt2_' + name+'.mp4'), writer=writer)