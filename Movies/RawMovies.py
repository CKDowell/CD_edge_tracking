# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 18:23:24 2025

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
from Utilities.utils_general import utils_general as ug
plt.rcParams['pdf.fonttype'] = 42 
from skimage import io
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
#%% load tiffs

planes = [3,4]
import cv2
fileroot = os.path.join(cxa.datadir,'registered',)
for i,p in enumerate(planes):
    fname = os.path.join(fileroot,cxa.name+'_slice'+str(p)+'.tif')
    im = io.imread(fname)
    if i==0:
        fullraw = np.zeros((im.shape[0],im.shape[1],im.shape[2],len(planes)))
    fullraw[:,:,:,i] = im

fullraw_mean = np.mean(fullraw,axis=3)
raw_prctile5 = np.percentile(fullraw_mean,5,axis=0)
raw_prctile97 = np.percentile(fullraw_mean,97,axis=0)
fullmean = (fullraw_mean-raw_prctile5)/(raw_prctile97-raw_prctile5)

planes = [1,2,3]
import cv2
fileroot = os.path.join(cxa.datadir,'registered',)
fname = os.path.join(fileroot,'ebmask.tiff')
ebm = io.imread(fname)
for i,p in enumerate(planes):
    fname = os.path.join(fileroot,cxa.name+'_slice'+str(p)+'.tif')
    im = io.imread(fname)
    if i==0:
        fullraw2 = np.zeros((im.shape[0],im.shape[1],im.shape[2],len(planes)))
        
    teb = ebm[:,:,p-1]==0
    fullraw2[:,:,:,i] = im*teb.astype('float')

fullraw2_mean = np.mean(fullraw2,axis=3)
raw_prctile5 = np.percentile(fullraw2_mean,5,axis=0)
raw_prctile97 = np.percentile(fullraw2_mean,97,axis=0)
fullmean2 = (fullraw2_mean-raw_prctile5)/(raw_prctile97-raw_prctile5)


#%% Movie
import matplotlib as mpl
from matplotlib.animation import ImageMagickFileWriter, ImageMagickWriter,FFMpegWriter
instrip = cxa.ft2['instrip'].to_numpy()
stripdiff = np.diff(instrip)
stripon = np.where(instrip>0)[0][0]

fig, axs = plt.subplots(figsize=(15,8))
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()

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
#ax = plt.subplot2grid((1, 3), (0, 0), colspan=1)

# Trajectory
ax2 = plt.subplot2grid((2, 2), (0,0),rowspan=2)
line, = ax2.plot([],[],lw=2,color='k')

# raw data eb
ax3 = plt.subplot2grid((2, 2), (0, 1))



ft2 = cxa.ft2

ins = ft2['instrip'].to_numpy()
jumps = ft2['jump'].to_numpy()
tt = cxa.pv2['relative_time'].to_numpy()
tim = np.linspace(tt[0],tt[-1],fullmean.shape[0])
inplume = ins>0
st  = np.where(inplume)[0][0]
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


im1 = ax3.imshow(fullmean[0,:,:].T,cmap='plasma',vmin=np.percentile(fullmean[:],20),vmax=np.percentile(fullmean[:],99))
#im2  = ax.imshow(fullraw2_mean[0,:,:].T,cmap='plasma',vmin=np.percentile(fullraw2_mean[:],10),vmax=np.percentile(fullraw2_mean[:],99))
ax3.set_xlim(4,fullraw_mean.shape[1]-4)
ax3.set_ylim(fullraw_mean.shape[2]-4,4)
#ax.set_xlim(4,fullraw_mean.shape[1]-4)
#ax.set_ylim(fullraw_mean.shape[2]-4,4)
ax3.set_xticks([])
ax3.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
def update(frame):
    t = tt[frame]
    tn = ug.find_nearest(tim,t)
    thisim = fullmean[tn,:,:].T
    im1.set_data(thisim)
    thisim2 = fullraw2_mean[tn,:,:].T
    #im2.set_data(thisim2)
    if frame>100:
        line.set_data(x[frame-100:frame], y[frame-100:frame])
    else:
        line.set_data(x[:frame], y[:frame])
    ax2.set_xlim(x[frame]-10,x[frame]+10)
    ax2.set_ylim(y[frame]-10, y[frame]+10)


anim = mpl.animation.FuncAnimation(fig, update, frames=np.arange(stripon-200,len(x)), interval=10)
plt.show()
writer = FFMpegWriter(fps=20)
savedir = "Y:\Data\FCI\FCI_summaries\FC2_maimon2"
path_to_convert = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe'
path_to_magick = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe'
anim.save(os.path.join(savedir,'EPG_phase_df_f' + name+'.avi'), writer=writer)


#%% Movie - black
import matplotlib as mpl
from matplotlib.animation import ImageMagickFileWriter, ImageMagickWriter,FFMpegWriter
instrip = cxa.ft2['instrip'].to_numpy()
stripdiff = np.diff(instrip)
stripon = np.where(instrip>0)[0][0]

fig, axs = plt.subplots(figsize=(15,8))
fig.patch.set_facecolor('black')
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()

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
#ax = plt.subplot2grid((1, 3), (0, 0), colspan=1)

# Trajectory
ax2 = plt.subplot2grid((2, 2), (0,0),rowspan=2)
line, = ax2.plot([],[],lw=2,color='w')
line2, = ax2.plot([],[],lw=2,color='r')
ax2.set_facecolor('black')
# raw data eb
ax3 = plt.subplot2grid((2, 2), (0, 1))
ax3.set_facecolor('black')


ft2 = cxa.ft2

ins = ft2['instrip'].to_numpy()
jumps = ft2['jump'].to_numpy()
tt = cxa.pv2['relative_time'].to_numpy()
tim = np.linspace(tt[0],tt[-1],fullmean.shape[0])
inplume = ins>0
st  = np.where(inplume)[0][0]
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


im1 = ax3.imshow(fullmean[0,:,:].T,cmap='plasma',vmin=np.percentile(fullmean[:],20),vmax=np.percentile(fullmean[:],99))
#im2  = ax.imshow(fullraw2_mean[0,:,:].T,cmap='plasma',vmin=np.percentile(fullraw2_mean[:],10),vmax=np.percentile(fullraw2_mean[:],99))
ax3.set_xlim(4,fullraw_mean.shape[1]-4)
ax3.set_ylim(fullraw_mean.shape[2]-4,4)
#ax.set_xlim(4,fullraw_mean.shape[1]-4)
#ax.set_ylim(fullraw_mean.shape[2]-4,4)
ax3.set_xticks([])
ax3.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
def update(frame):
    t = tt[frame]
    tn = ug.find_nearest(tim,t)
    thisim = fullmean[tn,:,:].T
    im1.set_data(thisim)
    thisim2 = fullraw2_mean[tn,:,:].T
    #im2.set_data(thisim2)
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
    if frame>100:
        line.set_data(x[frame-100:frame], y[frame-100:frame])
    else:
        line.set_data(x[:frame], y[:frame])
        
    ax2.set_xlim(x[frame]-10,x[frame]+10)
    ax2.set_ylim(y[frame]-10, y[frame]+10)
    

anim = mpl.animation.FuncAnimation(fig, update, frames=np.arange(stripon-200,len(x)), interval=10)
plt.show()
writer = FFMpegWriter(fps=20)
savedir = "Y:\Data\FCI\FCI_summaries\FC2_maimon2"
path_to_convert = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe'
path_to_magick = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe'
anim.save(os.path.join(savedir,'EPG_phase_df_f' + name+'_black.mp4'), writer=writer)
