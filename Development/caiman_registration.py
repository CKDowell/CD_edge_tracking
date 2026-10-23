# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 11:37:35 2025

@author: dowel
"""

#%% Multifly
import os
from analysis_funs.CX_registration_caiman import CX_registration_caiman as CX_cai
import multiprocessing
expdirs = [
   # r'Y:\Data\FCI\Hedwig\FB5A\260812\f1',
   # r'Y:\Data\FCI\Hedwig\hDeltaF_SS53683_sytGCaMP7f\260812\f1'
   #r'Y:\Data\FCI\Hedwig\PFNa_EPG\260814\f1'
   # r'Y:\Data\FCI\Hedwig\hDeltaF_SS53683_sytGCaMP7f\260813\f1',
   # r'Y:\Data\FCI\Hedwig\hDeltaF_SS53683_sytGCaMP7f\260813\f2'
   #r'Y:\Data\FCI\Hedwig\PFNa_EPG\260818\f2'
  # r'Y:\Data\FCI\Hedwig\hDeltaF_SS53683_sytGCaMP7f\260818\f1'
   # r'Y:\Data\FCI\Hedwig\hDeltaD_sytGCaMP7f\260819\f1',
   # r'Y:\Data\FCI\Hedwig\hDeltaD_sytGCaMP7f\260819\f2'
   
   # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260820\f1',
   # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260820\f2',
  # r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260824\f1'
  # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260825\f1'
   
   # r'Y:\Data\FCI\Hedwig\PFNpm_SS00191\260826\f1',
   # r'Y:\Data\FCI\Hedwig\hDeltaD_SS65202_sytGCaMP7f\260826\f1'
   #r'Y:\Data\FCI\Hedwig\FB5A_60D05\260831\f2'
   #r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260901\f1'
   # r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260911\f1',
   # r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260911\f2',
  # r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\260911\f1'
   #r'Y:\Data\FCI\Hedwig\Gr64f_hDeltaC_EPG_RCaMP\260914\f1'
   #r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260918\f1',
   
   # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f1',
   # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260922\f2'
   # r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\260922\f1',
   # r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\260923\f1'
  # r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\260923\f1'
  #r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\261008\f1'
  #r'Y:\Data\FCI\Hedwig\hDeltaE_sytGCaMP7f\261008\f1',
  #r'Y:\Data\FCI\Hedwig\hDeltaJ_Gr64f_EPG_RCaMP\261009\f1'
  #r'Y:\Data\FCI\Hedwig\PFNpm_SS00191\261013\f1'
  #r'Y:\Data\FCI\Hedwig\FC2_FB5A\261015\f1'
  r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\261020\f1',
  r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\261020\f2',
  r'Y:\Data\FCI\Hedwig\hDeltaJ_hDeltaB_EPG\261020\f3'
   ]

for e in expdirs:
    
    trial_folders = [name for name in os.listdir(e)
                        if os.path.isdir(os.path.join(e, name)) and "Trial" in name]
    for t in trial_folders:
        datadir = os.path.join(e,t)
        if __name__ == '__main__':
            
            multiprocessing.freeze_support()
            cxcai = CX_cai(datadir,dual_color=True)
            cxcai.one2other=False
            cxcai.chreg = 1
            cxcai.chmov = 2
            
            cxcai.register_rigid(params={'max_shifts': (4, 4),'max_deviation_rigid': 2,'pw_rigid':True})
            # cxcai.ex.mask_slice = {'All': [1,2,3,4]}
            # cxcai.ex.t_projection_mask_slice()
            

#%% Multifly


expdirs = [
     r'Y:\Data\FCI\Hedwig\hDeltaJ_Gr64f_EPG_RCaMP\261009\f1'
]
from analysis_funs.utilities import imaging as im
for e in expdirs:
    
    trial_folders = [name for name in os.listdir(e)
    if os.path.isdir(os.path.join(e, name)) and "Trial" in name]
    for t in trial_folders:
        datadir = os.path.join(e,t)
        d = datadir.split("\\")
        name = d[-3] + '_' + d[-2] + '_' + d[-1]
        ex = im.fly(name, datadir)
        ex.mask_slice = {'All': [1,2,3,4]}
        ex.t_projection_mask_slice()
#%% Single fly


# rootdir = r'Y:\Data\FCI\Hedwig\FC2_GCaMP8s_hDeltaJ_RCaMP3\261008\f1'
# import os
# from analysis_funs.CX_registration_caiman import CX_registration_caiman as CX_cai
# import multiprocessing
# for i in [3,4]:
#     datadir =os.path.join(rootdir,'Trial' +str(i))
#     d = datadir.split("\\")
#     name = d[-3] + '_' + d[-2] + '_' + d[-1]
#     #% Registration
#     # ex = im.fly(name, datadir)
#     # ex.register_all_images(overwrite=True)
#     if __name__ == '__main__':
        
#         multiprocessing.freeze_support()
#         cxcai = CX_cai(datadir,dual_color=True)
#         cxcai.one2other=False
#         cxcai.chreg = 1 
#         cxcai.chmov = 2
        
#         cxcai.register_rigid(params={'max_shifts': (4, 4),'max_deviation_rigid': 2,'pw_rigid':True})
#         # cxcai.ex.mask_slice = {'All': [1,2,3,4]}
#         # cxcai.ex.t_projection_mask_slice()
        
#%%
# datadir = r'Y:\Data\FCI\Hedwig\Tests\BleedThroughCheck\ACV pulses'
# d = datadir.split("\\")
# name = d[-3] + '_' + d[-2] + '_' + d[-1]
# #% Registration
# # ex = im.fly(name, datadir)
# # ex.register_all_images(overwrite=True)
# if __name__ == '__main__':
#     import multiprocessing
#     multiprocessing.freeze_support()
#     cxcai = CX_cai(datadir,dual_color=True)
#     cxcai.one2other=True
#     cxcai.chreg = 2 
#     cxcai.chmov = 1
#     cxcai.register_rigid(params={'max_shifts': (4, 4),'max_deviation_rigid': 2,'pw_rigid':True})
#%%
# from analysis_funs.utilities import imaging as im

# for i in [1,2,3]:
#     datadir =os.path.join(rootdir,'Trial' +str(i))
    # d = datadir.split("\\")
    # name = d[-3] + '_' + d[-2] + '_' + d[-1]
    # ex = im.fly(name, datadir)
    # ex.mask_slice = {'All': [1,2,3,4]}
    # ex.t_projection_mask_slice()
    
#%% Jules test

# datadir = r"D:\Tests\JulesTest\example1"
# import os
# from analysis_funs.CX_registration_caiman import CX_registration_caiman as CX_cai
# cxcai = CX_cai(datadir,dual_color=False)
# params = cxcai.default_params
# params['max_shifts'] = (20,20)
# cxcai.params = params


# cxcai.run_rigid_register(os.path.join(datadir,'20251222_Dmel_ChAT-sytGCaMP6f_f2-018 #2.tif'),plane=0)
# #%%
# from skimage import io,filters
# params['max_shifts'] = (30,30)
# stack = io.imread(os.path.join(datadir,'20251222_Dmel_ChAT-sytGCaMP6f_f2-018 #2.tif'))
# cxcai.tempfolder = r"D:\Tests\JulesTest\example1"
# cxcai.params = params
# mc = cxcai.run_rigid_stack(stack,tempname='Stack2')