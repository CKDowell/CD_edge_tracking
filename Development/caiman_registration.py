# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 11:37:35 2025

@author: dowel
"""

# rootdir = r"Y:\Data\FCI\Hedwig\hDeltaH_SS92512\251204\f2"
# import os
# from analysis_funs.CX_registration_caiman import CX_registration_caiman as CX_cai
# for i in [2,3,4,5]:
#     datadir =os.path.join(rootdir,'Trial' +str(i))
#     d = datadir.split("\\")
#     name = d[-3] + '_' + d[-2] + '_' + d[-1]
#     #% Registration
#     # ex = im.fly(name, datadir)
#     # ex.register_all_images(overwrite=True)
#     if __name__ == '__main__':
#         import multiprocessing
#         multiprocessing.freeze_support()
#         cxcai = CX_cai(datadir,dual_color=False)
#         cxcai.register_rigid(params={'max_shifts': (4, 4),'max_deviation_rigid': 2})
#         # cxcai.ex.mask_slice = {'All': [1,2,3,4]}
#         # cxcai.ex.t_projection_mask_slice()
        
#%%
from analysis_funs.utilities import imaging as im

for i in [2,3,4,5]:
    datadir =os.path.join(rootdir,'Trial' +str(i))
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    ex = im.fly(name, datadir)
    ex.mask_slice = {'All': [1,2,3,4]}
    ex.t_projection_mask_slice()
