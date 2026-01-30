# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 11:37:35 2025

@author: dowel
"""

rootdir = r"Y:\Data\FCI\Hedwig\hDeltaJ\260127\f1"
import os
from analysis_funs.CX_registration_caiman import CX_registration_caiman as CX_cai
for i in [2,3,4]:
    datadir =os.path.join(rootdir,'Trial' +str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    #% Registration
    # ex = im.fly(name, datadir)
    # ex.register_all_images(overwrite=True)
    if __name__ == '__main__':
        import multiprocessing
        multiprocessing.freeze_support()
        cxcai = CX_cai(datadir,dual_color=False)
        cxcai.one2other=False
        cxcai.chreg = 2 
        cxcai.chmov = 1
        
        cxcai.register_rigid(params={'max_shifts': (4, 4),'max_deviation_rigid': 2,'pw_rigid':True})
        # cxcai.ex.mask_slice = {'All': [1,2,3,4]}
        # cxcai.ex.t_projection_mask_slice()
        
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
from analysis_funs.utilities import imaging as im

for i in [1,2,3,4]:
    datadir =os.path.join(rootdir,'Trial' +str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    ex = im.fly(name, datadir)
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
    
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