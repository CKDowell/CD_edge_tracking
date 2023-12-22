# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:11:26 2023

@author: dowel
"""

#%%
import numpy as np
import pandas as pd
import src.utilities.funcs as fc
import os
import matplotlib.pyplot as plt
#%% MBON21_stimulation
meta_data = {'stim_type': 'plume',
    'ledONy': 300,
             'ledOffy':600,
             'LEDoutplume': True,
             'LEDinplume': False,
             'PlumeWidth': float(50),
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231214\\f1\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231214\\f1\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231214\\f2\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231214\\f2\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231214\\f2\Trial3"]
sdir = savedirs[0]

lname = os.listdir(sdir)
savepath = os.path.join(sdir,lname[0])
df = fc.read_log(savepath)
#%% 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    idx =idx
    return idx
def plot_plume(meta_data,df):
    x = pd.Series.to_numpy(df['ft_posx'])
    y = pd.Series.to_numpy(df['ft_posy'])
    
    #pon = pd.Series.to_numpy(df['mfc2_stpt']>0)
    pon = pd.Series.to_numpy(df['instrip']>0)
    pw = np.where(pon)
    x = x-x[pw[0][0]]
    y = y-y[pw[0][0]]
    plt.figure(figsize=(8,16))
    
    yrange = [min(y), max(y)]
    xrange = [min(x), max(x)]
    
    
    # Plt plume
    pi = np.pi
    psize =meta_data['PlumeWidth']
    pa = meta_data['PlumeAngle']
    xmplume = yrange[1]/np.tan(pi*(pa/180))
    xp = [-psize/2, yrange[1]*np.tan(pi*(pa/180))-psize/2,yrange[1]*np.tan(pi*(pa/180))+psize/2, psize/2,-psize/2]
    
    pan = meta_data['PlumeAngle']
    
    
    # Plt opto
    lo = meta_data['ledONy']
    loff = meta_data['ledOffy']
    yo = [lo,lo,loff,loff,lo]
    lin = meta_data['LEDinplume']
    lout = meta_data['LEDoutplume']
    ym = yrange[1]
    yplus = float(0)
    xsub_old =0
    while ym>1000:
        
        if lout and not lin:
            xo = [-500,500,500,-500, -500 ]
            plt.fill(xo,np.add(yo,yplus),color = [1,0.8,0.8])
            #plt.fill(np.multiply(-1,xo),np.add(yo,yplus),color = [1,0.8,0.8])
        yp = [yplus, yplus+1000, yplus+1000,yplus,yplus] 
        xsub = find_nearest(y,yplus)
        print(yplus)
        print(y[xsub])
        if yplus>0 :
            
            plt.fill(xp+x[xsub],yp,color =[0.8,0.8,0.8])
        
        else:
            plt.fill(xp,yp,color =[0.8,0.8,0.8])
        xsub_old = xsub    
        ym = ym-float(1000)
        yplus = yplus+float(1000)
        
    plt.ylim([0,1000])
    plt.xlim([-500,500])
    plt.plot(x,y,color='k')
    plt.scatter(x[pon],y[pon], color = [0.8, 0.8 ,0.2])
def plot_plume_simple(meta_data,df):
    
    x = pd.Series.to_numpy(df['ft_posx'])
    y = pd.Series.to_numpy(df['ft_posy'])
    x,y = fictrac_repair(x,y)
    s_type = meta_data['stim_type']
    plt.figure(figsize=(16,16))
    if s_type =='plume':
    
        #pon = pd.Series.to_numpy(df['mfc2_stpt']>0)
        pon = pd.Series.to_numpy(df['instrip']>0)
        pw = np.where(pon)
        x = x-x[pw[0][0]]
        y = y-y[pw[0][0]]
        
        
        yrange = [min(y), max(y)]
        xrange = [min(x), max(x)]
        xlm = np.max(np.abs(xrange))
        
        # Plt plume
        pi = np.pi
        psize =meta_data['PlumeWidth']
        pa = meta_data['PlumeAngle']
        xmplume = yrange[1]/np.tan(pi*(pa/180))
        xp = [-psize/2, yrange[1]*np.tan(pi*(pa/180))-psize/2,yrange[1]*np.tan(pi*(pa/180))+psize/2, psize/2,-psize/2]
        
        pan = meta_data['PlumeAngle']
        
        yp = [yrange[0], yrange[1], yrange[1],yrange[0],yrange[0]] 
        
        xo = [-xlm,xlm,xlm,-xlm, -xlm ]
        if meta_data['ledONy']=='all':
            lo = yrange[0]
        else:
            lo = meta_data['ledONy']
        
        if meta_data['ledOffy']=='all':
            loff = yrange[1]
            
        else:
            loff = meta_data['ledOffy']
        yo = [lo,lo,loff,loff,lo]
        
        plt.fill(xo,yo,color = [1,0.8,0.8])
        
        if loff<yrange[1]:
            while loff<yrange[1]:
                loff = loff+1000
                lo = lo+1000
                yo = [lo,lo,loff,loff,lo]
                plt.fill(xo,yo,color = [1,0.8,0.8])
        
        plt.fill(xp,yp,color =[0.8,0.8,0.8])
        # Add in extra for repeated trials
        plt.plot(x[pw[0][0]:],y[pw[0][0]:],color='k')
        plt.plot(x[0:pw[0][0]],y[0:pw[0][0]],color=[0.5,0.5,0.5])
        #plt.scatter(x[pon],y[pon], color = [0.8, 0.8 ,0.2])
        yxlm = np.max(np.abs(np.append(yrange,xrange)))
        ymn = np.mean(yrange)
        #plt.ylim([ymn-(yxlm/2), ymn+(yxlm/2)])
        #plt.xlim([-1*(yxlm/2), yxlm/2])
    elif s_type == 'pulse':
        led = df['led1_stpt']==0
        plt.scatter(x[led],y[led],color=[1,0.8,0.8])
        plt.plot(x,y,color='k')
    plt.gca().set_aspect('equal')
    plt.show()
def light_pulse_pre_post(meta_data,df):
    plt.figure(figsize=(10,10))
    x = pd.Series.to_numpy(df['ft_posx'])
    y = pd.Series.to_numpy(df['ft_posy'])
    x,y = fictrac_repair(x,y)
    t = get_time(df)
    
    led = df['led1_stpt']
    led_on = np.diff(led)<0 
    led_off = np.diff(led)>0 
    lo_dx = [i+1 for i,ir in enumerate(led_on) if ir]
    loff_dx = [i+1 for i,ir in enumerate(led_off) if ir]
    if len(loff_dx)<len(lo_dx):
        lo_dx = lo_dx[:-1]
    tbef = 2 
    tdx = np.sum(t<tbef)
    ymax = 0
    for i,on in enumerate(lo_dx):
        st = on-tdx
        st_o = loff_dx[i]
        y_b = y[st:on]
        x_b = x[st:on]
        x_vec = x_b[-1]-x_b[0]
        y_vec = y_b[-1]-y_b[0]
        theta = -np.arctan(y_vec/x_vec)-np.pi
        hyp = np.sqrt(x_vec**2+y_vec**2)
        cos_thet = np.cos(theta)
        sin_thet = np.sin(theta)
        rotmat = np.array([[cos_thet, -sin_thet],[sin_thet, cos_thet]])
        plt_x = x[st:st_o]-x[on]
        plt_y = y[st:st_o]-y[on]
        xymat = np.array([plt_x,plt_y])
        
        rot_xy = np.matmul(rotmat,xymat)
        # need to flip x axis if negative
        if rot_xy[0,0]>0:
            rot_xy[0,:] = -rot_xy[0,:]
        plt.plot(rot_xy[0,:tdx],rot_xy[1,:tdx],color= [0.8,0.8,0.8])
        plt.plot(rot_xy[0,tdx+1:],rot_xy[1,tdx+1:],color='k')
        ymax = np.max(np.abs(np.append(rot_xy[1,:],ymax)))   
    plt.ylim([-ymax,ymax])
    plt.gca().set_aspect('equal')
    plt.show()
def get_time(df):
    t = pd.Series.to_numpy(df['timestamp'])
    t = np.array(t,dtype='str')
    t_real = np.empty(len(t),dtype=float)
    for i,it in enumerate(t):
        tspl = it.split('T')
        tspl2 = tspl[1].split(':')
        t_real[i] = float(tspl2[0])*3600+float(tspl2[1])*60+float(tspl2[2])
    t_real = t_real-t_real[0]
    return t_real
def fictrac_repair(x,y):
    dx = np.abs(np.diff(x))
    dy = np.abs(np.diff(y))
    lrgx = dx>5 
    lrgy = dy>5 
    bth = np.logical_or(lrgx, lrgy)
    
    fixdx =[i+1 for i,b in enumerate(bth) if b]
    for i,f in enumerate(fixdx):
        
        x[f:] =x[f:]- (x[f]-x[f-1])
        
        y[f:] = y[f:]- (y[f]-y[f-1])
    return x, y
#%% Plot 1st run through
plt.close('all')
for sdir in savedirs:
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    plot_plume(meta_data,df)
#%% No odour stim outside
plt.close('all')
meta_data = {'stim_type': 'plume',
    'ledONy': 'all',
             'ledOffy':'all',
             'LEDoutplume': True,
             'LEDinplume': False,
             'PlumeWidth': float(50),
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON21_ACV_stim_border\\231214\\f2\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation_no_odour\\231214\\f2\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation_no_odour\\231215\\f2\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation_no_odour\\231215\\f2\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation_no_odour\\231215\\f3\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation_no_odour\\231215\\f3\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation_no_odour\\231215\\f4\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation_no_odour\\231215\\f4\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation_no_odour\\231215\\f5\Trial1_30mm",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation_no_odour\\231215\\f5\Trial2_10mm",
            ]
plumewidths = [50,50,50,50,50,50,50,50,30,10]
pltsavedir = "Y:\Data\Optogenetics\MBONs\MBON21_stimulation_no_odour\SummaryFigures"
for i in range(len(savedirs)):
    sdir = savedirs[i]

    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    meta_data['PlumeWidth'] = plumewidths[i]
    plot_plume_simple(meta_data,df)
    snames = sdir.split('\\')
    plt.title(snames[-3] + ' ' + snames[-2] + ' ' +snames[-1])
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.png'
    plt.savefig(os.path.join(pltsavedir,savename))
#%% ACV and stim outside without shifting plume location
plt.close('all')
meta_data = {'stim_type': 'plume',
    'ledONy': 300,
             'ledOffy':600,
             'LEDoutplume': True,
             'LEDinplume': False,
             'PlumeWidth': float(50),
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231215\\f2\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231215\\f3\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231215\\f4\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_stimulation\\231215\\f5\Trial1",
            ]
for i in range(len(savedirs)):
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    plot_plume_simple(meta_data,df)
#%% MBON 21 stimulaton pulses
plt.close('all')
meta_data = {'stim_type': 'pulse',
    'ledONy': 0,
             'ledOffy':0,
             'LEDoutplume': False,
             'LEDinplume': False,
             'PlumeWidth': False,
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON21_light_pulses\\231219\\f1\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON21_light_pulses\\231219\\f2\Trial1"]
for i in range(len(savedirs)):
    print(i)
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    plot_plume_simple(meta_data,df)
    #light_pulse_pre_post(meta_data,df)
#%% MBON 33 stimulation pulses
plt.close('all')
meta_data = {'stim_type': 'pulse',
    'ledONy': 0,
             'ledOffy':0,
             'LEDoutplume': False,
             'LEDinplume': False,
             'PlumeWidth': False,
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231219\\f4\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231219\\f4\Trial2_15s",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231219\\f5\Trial1_15s",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231220\\f1\Trial1",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231220\\f1\Trial2",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231220\\f2\Trial1"]
pltsavedir = "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\SummaryFigures"
for i in range(len(savedirs)):
    print(i)
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    light_pulse_pre_post(meta_data,df)
    snames = sdir.split('\\')
    plt.title(snames[-3] + ' ' + snames[-2] + ' ' +snames[-1])
    savename = snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.png'
    plt.savefig(os.path.join(pltsavedir,savename))
    plot_plume_simple(meta_data,df)
    savename = 'Traj_' + snames[-3] + '_' + snames[-2] + '_' +snames[-1] + '.png'
    plt.savefig(os.path.join(pltsavedir,savename))
#%% MBON 33 stimulation after edge trackingplt.close('all')
plt.close('all')
meta_data = {'stim_type': 'pulse',
    'ledONy': 0,
             'ledOffy':0,
             'LEDoutplume': False,
             'LEDinplume': False,
             'PlumeWidth': False,
             'PlumeAngle': 0,
             }
savedirs = ["Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231220\\f1\Trial3_post_plume",
            "Y:\Data\Optogenetics\MBONs\MBON33_light_pulses\\231220\\f2\Trial2_post_plume"
            ]
for i in range(len(savedirs)):
    print(i)
    sdir = savedirs[i]
    lname = os.listdir(sdir)
    savepath = os.path.join(sdir,lname[0])
    df = fc.read_log(savepath)
    plot_plume_simple(meta_data,df)
    #light_pulse_pre_post(meta_data,df)