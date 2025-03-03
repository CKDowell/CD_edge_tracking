# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:09:14 2024

@author: dowel
"""

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 

#%%
plt.close('all')
meta_data = {'stim_type': 'plume',
              'act_inhib':'act',
    'ledOny': -float(50)/2,
              'ledOffy':float(50)/2,
              'ledOnx': 10,
              'ledOffx': 30,
              'LEDoutplume': False,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 90,
              'RepeatInterval':250
              }
savedirs = ["Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal\\240118\\f2\Trial1",
        "Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal\\240121\\f1\Trial1",
        "Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal\\240121\\f2\Trial1",
        "Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal\\240121\\f3\Trial1",
        "Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal\\240124\\f1\Trial1",
        "Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal\\240124\\f2\Trial1",
        "Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal\\240124\\f2\Trial2",
        "Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal\\240124\\f4\Trial1",
        ]
figdir = r'Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal\SummaryFigures'
for i in range(len(savedirs)):
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_horizontal(meta_data,df)
    snames = sdir.split('\\')
    plt.savefig(os.path.join(figdir,snames[-3]+snames[-2]+snames[-1] +'.png'))
    plt.savefig(os.path.join(figdir,snames[-3]+snames[-2]+snames[-1] +'.pdf'))
    
#%% Horizontal delay

plt.close('all')
meta_data = {'stim_type': 'plume',
              'act_inhib':'inhib',
    'ledOny': -float(50)/2,
              'ledOffy':float(50)/2,
              'ledOnx': 10,
              'ledOffx': 30,
              'LEDoutplume': False,
              'LEDinplume': True,
              'PlumeWidth': float(50),
              'PlumeAngle': 90,
              'RepeatInterval':250
              }
savedirs = [
    #"Y:\\Data\\Optogenetics\\Gr64-f\\Gr64f_HorizontalDelay\\240131\\f2\\Trial1",#Stimulation set up wrong
    "Y:\\Data\\Optogenetics\\Gr64-f\\Gr64f_HorizontalDelay\\240202\\f1\\Trial1",
    "Y:\\Data\\Optogenetics\\Gr64-f\\Gr64f_HorizontalDelay\\240208\\f1\\Trial1",
    "Y:\\Data\\Optogenetics\\Gr64-f\\Gr64f_HorizontalDelay\\240208\\f2\\Trial1",
        
        ]    

for i in range(len(savedirs)):
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_horizontal(meta_data,df)
#%%    
plt.close('all')
meta_data = {'stim_type': 'plume',
              'act_inhib':'act',
    'ledOny': float(50)/2,
              'ledOffy':float(50)/2+10,
              'ledOnx': 10,
              'ledOffx': 30,
              'LEDoutplume': True,
              'LEDinplume': False,
              'PlumeWidth': float(50),
              'PlumeAngle': 90,
              'RepeatInterval':250
              }
savedirs = ["Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal_Outside\\240124\\f4\Trial1",
            "Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal_Outside\\240131\\f1\Trial1",
            "Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal_Outside\\240131\\f2\Trial1"
        
        ]
for i in range(len(savedirs)):
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_horizontal(meta_data,df)
    
#%%    
import matplotlib.animation as animation
from functools import partial
def plume_horizontal_movie(meta_data,df):
    global x,y, chunk_size
    x = pd.Series.to_numpy(df['ft_posx'])
    y = pd.Series.to_numpy(df['ft_posy'])
    
    
    #x,y = self.fictrac_repair(x,y)
    
    pon = pd.Series.to_numpy(df['instrip']>0)
    pw = np.where(pon)
    x = x-x[pw[0][0]]
    y = y-y[pw[0][0]]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal')
    chunk_size = 100
    line1, = ax.plot([],[],'ro')
    
    def update_fig(frame):
        x1 = x[frame:(frame+chunk_size)]
        y1 = y[frame:(frame+chunk_size)]
        line1.set_data(x1,y1)
        return line1,
    print(x)
    ani = animation.FuncAnimation(fig,update_fig,
                                  frames = np.linspace(0,len(x),len(x)-1,dtype=int),
                                  blit = True)
    plt.show()
                                  
plume_horizontal_movie(meta_data,df)          
#%%
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from functools import partial


# def animate(frame,)

# animation = FuncAnimation(fig,animate,total_number_of_frames, fargs=[image_eb,image_fb, line1, line2, line3,], interval=interval)
# plt.show()
# # save_name = os.path.join(ex.figure_folder, 'movie.avi')
# animation.save(save_name, writer='ffmpeg', codec='mjpeg')
# plt.style.use('default')
    #writer = animation.writers['ffmpeg'](fps=30)

    #ani.save('demo.mp4',writer=writer,dpi=dpi)
#plume_horizontal_movie(meta_data,df)
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Your specific x and y values
x_values = pd.Series.to_numpy(df['ft_posx'])
y_values = pd.Series.to_numpy(df['ft_posy'])

# Create initial scatter plot
fig, ax = plt.subplots()
line, = ax.plot([], [],color='k')  # Empty scatter plot

# Set axis limits
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)  # Adjust the y-axis limits based on your data

# Animation update function
def update(frame):
    # Update the scatter plot with the current x and y values
    line.set_data(np.column_stack((x_values[:frame], y_values[:frame])))

# Create animation
animation = FuncAnimation(fig, update, frames=len(x_values), interval=100)

# Save the animation to a file (e.g., GIF or MP4)
#animation.save('sequence_animation.gif', writer='imagemagick')
#%% 
datadir = "Y:\Data\Optogenetics\Gr64-f\Gr64af_Horizontal\\240121\\f3\Trial1"
lname = os.listdir(datadir)
savepath = os.path.join(datadir,lname[0])
df = fc.read_log(savepath)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg'
import networkx as nx

#mpl.use("TkAgg") 
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import ImageMagickFileWriter, ImageMagickWriter,FFMpegWriter
# Your specific x and y values
x = pd.Series.to_numpy(df['ft_posx'])
y = pd.Series.to_numpy(df['ft_posy'])
z = pd.Series.to_numpy(df['led1_stpt'])

pon = pd.Series.to_numpy(df['instrip']>0)
pw = np.where(pon)
x = x-x[pw[0][0]]
y = y-y[pw[0][0]]
x = x[pw[0][0]:]
y = y[pw[0][0]:]
z = z[pw[0][0]:]
# Create initial line plot

fig, ax = plt.subplots(figsize=(20,20))
line2, = ax.plot([],[],lw=2,color=[0.2,0.2,0.2])
line, = ax.plot([], [], lw=2,color=[0.2,0.4,1])  # Empty line plot with line width specified
sc = ax.scatter([],[],color='red',s=100,zorder=10)
plt.box('False')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
pa = meta_data['PlumeWidth']
yrange = [min(y), max(y)]
xrange = [min(x), max(x)]
x_plm = [xrange[0], xrange[0], xrange[1],xrange[1]]
y_plm = [-pa/2, pa/2, pa/2, -pa/2]
plt.fill(x_plm,y_plm,color =[0.8,0.8,0.8])
x_on = meta_data['ledOnx']
x_off = meta_data['ledOffx']

y_on = meta_data['ledOny']
y_off= meta_data['ledOffy']
y_stm = [y_on, y_off, y_off,y_on]

rep_int = meta_data['RepeatInterval']

a_s = meta_data['act_inhib']
if a_s=='act':
    led_colour = [1,0.8,0.8]
elif a_s=='inhib':
    led_colour = [0.8, 1, 0.8]
if xrange[0]<0:
    
    xr = -np.arange(0,np.abs(xrange[0]),rep_int)
    print(xr)
    for i in xr:
        
        x_stm = [i-x_on, i-x_on,i-x_off,i-x_off]
        ax.fill(x_stm,y_stm,color=led_colour)
if xrange[1]>0:
    
    xr = np.arange(0,np.abs(xrange[1]),rep_int)
    print(xr)
    for i in xr:
        
        x_stm = [i+x_on, i+x_on,i+x_off,i+x_off]
        ax.fill(x_stm,y_stm,color=led_colour)

ax.set_aspect('equal')
# Set axis limits
ax.set_xlim(xrange[0], xrange[1])
ax.set_ylim(yrange[0], yrange[1])  # Adjust the y-axis limits based on your data

# Animation update function
def update(frame):
    # Update the line plot with the current x and y values
    line2.set_data(x[:frame], y[:frame])
    if frame>100:
        line.set_data(x[frame-100:frame], y[frame-100:frame])
    else:
        line.set_data(x[:frame], y[:frame])
    
    if z[frame]<1:
        sc.set_offsets(np.column_stack((x[frame],y[frame])))
    
# Create animation
animation = mpl.animation.FuncAnimation(fig, update, frames=len(x), interval=0.01)
savedir = "Y:\Data\Optogenetics\\Gr64-f\\Gr64af_Horizontal_Outside\\SummaryFigures"
path_to_convert = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe'
path_to_magick = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe'
#writer = ImageMagickWriter(fps=100, metadata=dict(artist='Me'), bitrate=1800)
writer = FFMpegWriter(fps=300)
#writer.program = path_to_magick

animation.save(os.path.join(savedir,'Outisde_Stim_fly1_better.avi'), writer=writer)

plt.show()         
#%%
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'k-o')

plt.xlim(-5, 5)
plt.ylim(-5, 5)

x0, y0 = 0, 0

with writer.saving(fig, "writer_test.mp4", 100):
    for i in range(100):
        x0 += 0.1 * np.random.randn()
        y0 += 0.1 * np.random.randn()
        l.set_data(x0, y0)
        writer.grab_frame()