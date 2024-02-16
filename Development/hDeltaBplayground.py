# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:16:34 2024

@author: dowel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
#%%
datadirs = ["Y:\Data\FCI\AndyData\hdb\\20220329_Fly2_007\processed",
            "Y:\Data\FCI\AndyData\hdb\\20220329_Fly2_008\processed",
            "Y:\Data\FCI\AndyData\hdb\\20220405_Fly3_001\processed",
            "Y:\Data\FCI\AndyData\hdb\\20220406_Fly2_001\processed"]
filename = datadirs[3]+'\postprocessing.h5'
data = h5py.File(filename,'r')
#data = pd.read_hdf(filename)
ft = data['ft']
ft2  =data['ft2']
ix = data['ix']
pv2 = data['pv2']
#%% 
from analysis_funs.FSB_imaging import hdeltac, hdc_example
h = hdeltac(celltype='hdeltab')
#h.register() works
#h.processing()
df = h.sheet
d = df.iloc[7].to_dict()
datafolder = "Y:\Data\FCI\AndyData\hdb"
ex = hdc_example(d,datafolder)

ex.save_all_info_eb_fb()
data = ex.fetch_all_info_eb_fb()
ex.plot_FB_heatmap_heading()
#%%
im_names = np.array(pv2['block0_items'])
im_vals = np.array(pv2['block0_values'])
ft_names = np.array(ft2['block0_items'])
ft_vals = np.array(ft2['block0_values'])
I = im_vals[:,1:16]
plt.imshow(I,aspect='auto',interpolation='none',cmap='coolwarm')
plt.show()
plt.Figure()
plt.imshow(im_vals[:,17:32],aspect='auto',interpolation='none',cmap='coolwarm')