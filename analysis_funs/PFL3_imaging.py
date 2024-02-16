# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:02:41 2024

@author: dowel
"""

#%% 

# Notebook acts as a processing pipeline for PFL3 neurons. 
# Protocerebral bridge only
#%%
import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
#%% Imaging test for PFL3 neurons


datadir = "Y:\Data\FCI\Hedwig\\SS82335\\240207\\f2\\Trial3"
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
ex = im.fly(name, datadir)
#%% Registration

ex.register_all_images(overwrite=True)
ex.z_projection()
#%% Save pre processing and post-processing data
# edited to remove .dat, not sure what information is needed from dat that was not saved
ex.save_preprocessing()

# Work on below - there may be an issue with how the signal is received. 
# Looks like it may not have been recorded for the entire exp
ex.save_postprocessing()
#%%
pv2,ft,ft2,ix = ex.load_postprocessing()
#%% PB masks for ROI drawing
ex.mask_slice = {'PB': [1,2,3]}
ex.t_projection_mask_slice()
#%% Use MatLab Gui to draw ROIs
#%% Get PB glomerulous 
pb = im.PB2(name,datadir,[1,2,3])
glms = pb.get_gloms()
#%% Check log vs dat files to see what is needed
dat_path = "Y:\Data\FCI\Test_dat_log\\240207\\f2\\Trial2\\fictrac-20240207_143243.dat"
log_path = "Y:\Data\FCI\Test_dat_log\\240207\\f2\\Trial2\\02072024-143249.log"
names = [
      'frame',
      'del_rot_cam_x',
      'del_rot_cam_y',
      'del_rot_cam_z',
      'del_rot_error',
      'df_pitch',
      'df_roll',
      'df_yaw',
      'abs_rot_cam_x',
      'abs_rot_cam_y',
      'abs_rot_cam_z',
      'abs_rot_lab_x',
      'abs_rot_lab_y',
      'abs_rot_lab_z',
      'ft_posx',
      'ft_posy',
      'ft_heading',
      'ft_movement_dir',
      'ft_speed',
      'forward_motion',
      'side_motion',
      'timestamp',
      'sequence_counter',
      'delta_timestep',
      'alt_timestep'
]
df1 = pd.read_table(dat_path, delimiter='[,]', names = names, engine='python')
df1.ft_posx = -3*df1.ft_posx # flip x and y for mirror inversion
df1.ft_posy = -3*df1.ft_posy
df1.ft_speed = 3*df1.ft_speed
df1['seconds'] = (df1.timestamp-df1.timestamp.iloc[0])/1000


df = pd.read_table(log_path, delimiter='[,]', engine='python')
#split timestamp and motor into separate columns
new = df["timestamp -- motor_step_command"].str.split("--", n = 1, expand = True)
df["timestamp"]= new[0]
df["motor_step_command"]=new[1]
df.drop(columns=["timestamp -- motor_step_command"], inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format="%m/%d/%Y-%H:%M:%S.%f ")
df['seconds'] = 3600*df.timestamp.dt.hour+60*df.timestamp.dt.minute+df.timestamp.dt.second+10**-6*df.timestamp.dt.microsecond

# motor command sent to arduino as string, need to convert to numeric
df['motor_step_command'] = pd.to_numeric(df.motor_step_command)

# CAUTION: invert x. Used to correct for mirror inversion, y already flipped in voe
df['ft_posx'] = -df['ft_posx']

#calculate when fly is in strip
df['instrip'] = np.where(df['mfc2_stpt']>0.0, True, False)



