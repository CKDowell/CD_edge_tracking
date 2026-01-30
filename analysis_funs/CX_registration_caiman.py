# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:54:35 2025

@author: dowel

This class is a wrapper for image registration using normcorre from Caiman. This differs
from the built in pipeline to CX and corrects for large drifts better. To use it requires
caiman is installed, and probably best run from its own environment since the installed
version I have runs on python 3.10



"""
import os

from analysis_funs.CX_imaging import CX
from src.utilities import imaging as im

import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from skimage import io,filters
import numpy as np
from analysis_funs.utilities import funcs as fn
#%%
class CX_registration_caiman:
    def __init__(self,datadir,**kwargs):
        d = datadir.split("\\")
        name = d[-3] + '_' + d[-2] + '_' + d[-1]
        self.ex = im.fly(name,datadir,**kwargs)
        self.default_params = {'max_shifts': (6, 6),  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
        'strides': (48, 48),  # create a new patch every x pixels for pw-rigid correction
        'overlaps': (24, 24),  # overlap between patches (size of patch strides+overlaps)
        'max_deviation_rigid': 3,   # maximum deviation allowed for patch with respect to rigid shifts
        'pw_rigid': False,  # flag for performing rigid or piecewise rigid motion correction
        'shifts_opencv': True,  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
        'border_nan': 'copy'}  # replicate values along the boundary (if True, fill in with NaN)
        self.temp_folder = r'D:\FCI\reg_temporary_data'
        self.rigid_out = {}
        self.one2other=False
    def register_rigid(self,params=[]):
        new_params = self.default_params.copy()
        if len(params)>0:
            for p in params:
                new_params[p] = params[p]
        self.params= new_params
        slice_stacks = self.ex.split_files()
        
        cm.cluster.stop_server()
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='multiprocessing',
            n_processes=None,
            single_thread=False
        )
        try:
            for i in np.arange(1, len(slice_stacks)+1):
                print('stack:', i)
                files = slice_stacks[i]
                files.sort()
                
                if hasattr(self.ex, 'split'):
                    for j, range in enumerate(self.ex.split):
                        tif_name = os.path.join(self.ex.regfol, self.cx.name+'_slice'+str(i)+'_split'+str(j)+'.tif')
                        pickle_name = os.path.join(self.ex.regfol, self.cx.name+'_slice'+str(i)+'_split'+str(j)+'.pickle')
                        files_sub = files[(range[0]-1):(range[1]-1)]
                        reg_results,registered_blurred,mc = self.run_rigid_register(files_sub,i,dview)
                        io.imsave(tif_name, registered_blurred, plugin='tifffile')
                        fn.save_obj(reg_results, pickle_name)
                elif self.ex.dual_color:
                    # need to create options for registering one color based on another, or registering both separately
                    Ch1 = [f for f in files if 'Ch1' in f]
                    Ch2 = [f for f in files if 'Ch2' in f]
                    Chns = [Ch1, Ch2]
                    
                    if self.one2other:
                        
                        reg_results,registered_blurred,registered_blurred2 = self.run_rigid_1_to_other(Chns[self.chreg-1],Chns[self.chmov-1],i,self.chreg,self.chmov,dview)
                        
                        tif_name = os.path.join(self.ex.regfol, self.ex.name+'_Ch'+str(self.chreg)+'_slice'+str(i)+'.tif')
                        io.imsave(tif_name, registered_blurred, plugin='tifffile')
                        pickle_name = os.path.join(self.ex.regfol, self.ex.name+'_Ch'+str(self.chreg)+'_slice'+str(i)+'.pickle')
                        tif_name = os.path.join(self.ex.regfol, self.ex.name+'_Ch'+str(self.chmov)+'_slice'+str(i)+'.tif')
                        io.imsave(tif_name, registered_blurred2, plugin='tifffile')
                        pickle_name = os.path.join(self.ex.regfol, self.ex.name+'_Ch'+str(self.chmov)+'_slice'+str(i)+'.pickle')
                    else:
    
                        for ch in [1,2]:
                            tif_name = os.path.join(self.ex.regfol, self.ex.name+'_Ch'+str(ch)+'_slice'+str(i)+'.tif')
                            pickle_name = os.path.join(self.ex.regfol, self.ex.name+'_Ch'+str(ch)+'_slice'+str(i)+'.pickle')
                            reg_results,registered_blurred,mc = self.run_rigid_register(Chns[ch-1],i,dview,ch=[ch])
                            
                            #reg_results, registered_blurred = self.register_image_block(Chns[ch-1], ini_reg_frames=100,two_scale = True,scale=0.5)
                            io.imsave(tif_name, registered_blurred, plugin='tifffile')
                            fn.save_obj(reg_results, pickle_name)
                elif self.ex.dual_color_old:
                    # for imaging on Lyndon where the channel names are 2 and 3 instead of 1 and 2
                    Ch2 = [f for f in files if 'Ch2' in f]
                    Ch3 = [f for f in files if 'Ch3' in f]
                    Chns = [Ch2, Ch3]
                    for ch in [2,3]:
                        tif_name = os.path.join(self.ex.regfol, self.ex.name+'_Ch'+str(ch)+'_slice'+str(i)+'.tif')
                        pickle_name = os.path.join(self.ex.regfol, self.ex.name+'_Ch'+str(ch)+'_slice'+str(i)+'.pickle')
                        reg_results,registered_blurred,mc = self.run_rigid_register(Chns[ch-1],i,dview,ch=[ch])
                        #reg_results, registered_blurred = self.register_image_block(Chns[ch-2], ini_reg_frames=100,two_scale = True,scale=0.5)
                        io.imsave(tif_name, registered_blurred, plugin='tifffile')
                        fn.save_obj(reg_results, pickle_name)
                else:
                    tif_name = os.path.join(self.ex.regfol, self.ex.name+'_slice'+str(i)+'.tif')
                    pickle_name = os.path.join(self.ex.regfol, self.ex.name+'_slice'+str(i)+'.pickle')
                    reg_results,registered_blurred,mc = self.run_rigid_register(files,i,dview)
                    #reg_results, registered_blurred = self.register_image_block(files, ini_reg_frames=100)
                    io.imsave(tif_name, registered_blurred, plugin='tifffile')
                    fn.save_obj(reg_results, pickle_name)
        finally:
            if dview is not None:
                dview.terminate()
            cm.cluster.stop_server()
    def run_rigid_register(self,files,plane,dview,ch=[]):
        
        params = self.params   
        
        
        original = self.load_stack(files,plane)
        
        registered = np.zeros(original.shape)
        registered_blurred = np.zeros(original.shape)
        
        if len(ch)==0:
            tempname = os.path.join(self.temp_folder, self.ex.name+'pre_register_slice'+str(plane)+'.tif')
        else:
            tempname = os.path.join(self.temp_folder, self.ex.name+'_Ch'+str(ch[0])+'pre_register_slice'+str(plane)+'.tif')
        io.imsave(tempname,original,plugin='tifffile')
        
        
                
        mc  = MotionCorrect(tempname,dview=dview,
                            max_shifts= params['max_shifts'],strides = params['strides'],
                            overlaps=params['overlaps'], max_deviation_rigid=params['max_deviation_rigid'],
                            shifts_opencv =params['shifts_opencv'],nonneg_movie = True,
                            border_nan=params['border_nan'])
        
        mc.motion_correct(save_movie=True)
        registered = cm.load(mc.mmap_file)
        self.rigid_out.update({'plane'+str(plane):mc.shifts_rig})
        
        if len(ch)==0:
            tif_name = os.path.join(self.temp_folder, self.ex.name+'_slice'+str(plane)+'.tif')
        else:
            tif_name = os.path.join(self.temp_folder, self.ex.name+'_Ch'+str(ch[0])+'_slice'+str(plane)+'.tif')
            
        io.imsave(tif_name,registered,plugin='tifffile') # save in temp folder for easy comparison
            
        for i, frame in enumerate(registered):
            registered_blurred[i] = filters.gaussian(frame, 1)
        reg_results = {'rigid_shift':mc.shifts_rig}
        return reg_results, registered_blurred,mc
    def load_stack(self,files,plane):
        f = files
        f1 = f[0]
        f1s = f1.split('.')
        chn = int(f1s[0][-1])-1
        images = io.imread_collection(f)
        # CD edit: io.concatenate does not work because tiffs are loaded in 
        # a strange way. This is a work around
        try :
            print(chn)
            all_images = [image[np.newaxis, ...] for image in images]
            all_images2 = all_images[1:]
            print(np.shape(all_images[0]))
            
            if len(np.shape(all_images[0]))==4:
                extra_im = all_images[0][0,chn,:,:]
            elif len(np.shape(all_images[0]))==5:
                extra_im = all_images[0][0,0,chn,:,:]
            elif len(np.shape(all_images[0]))==6:
                 extra_im = all_images[0][0,0,chn,plane-1,:,:]
            extra_im = extra_im[np.newaxis,...]
            all_images2.append(extra_im)
            stack = np.concatenate(all_images2)
        except: # CD edit -reordeded this since below actually takes some time
            print(chn)
            stack = io.concatenate_images(images)
        return stack
    def run_rigid_1_to_other(self,files_Chref,files_Chmov,plane,ref_chan,mov_chan,dview):
        reg_results, registered_blurred,mc = self.run_rigid_register(files_Chref,plane,dview,ch=[ref_chan])
        
        stack = self.load_stack(files_Chmov,plane)
        
        
        tempname = os.path.join(self.temp_folder, self.ex.name+'_Ch'+str(mov_chan)+'pre_register_slice'+str(plane)+'.tif')
        io.imsave(tempname,stack,plugin='tifffile')
        
        
        second_memmap = mc.apply_shifts_movie(tempname,save_memmap=True,
                                              save_base_name=os.path.join(self.temp_folder,self.ex.name+'_Ch'+str(mov_chan)+'_slice'+str(plane)))
        
        registered = cm.load(second_memmap)
        self.rigid_out.update({'plane'+str(plane):mc.shifts_rig})
        
        
        tif_name = os.path.join(self.temp_folder, self.ex.name+'_Ch'+str(mov_chan)+'_slice'+str(plane)+'.tif')
            
        io.imsave(tif_name,registered,plugin='tifffile') # save in temp folder for easy comparison
        registered_blurred2 = np.zeros(stack.shape)
        for i, frame in enumerate(registered):
            registered_blurred2[i] = filters.gaussian(frame, 1)
        reg_results = {'rigid_shift':mc.shifts_rig}
        
        return reg_results, registered_blurred,registered_blurred2
    
    
    def run_rigid_stack(self,stack,tempname='Stack1'):
        tempsave = os.path.join(self.tempfolder,tempname+'.tif')
        io.imsave(tempsave,stack,plugin='tifffile')
        params= self.params
        try:
            # If a cluster already exists, stop it
            
            cm.stop_server(dview=dview)
            dview.terminate()
        except Exception:
            pass
        
        # Now safely start a fresh one
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='multiprocessing',
            n_processes=None,
            single_thread=False
        )
        
        
                
                
        mc  = MotionCorrect(tempsave,dview=dview,
                            max_shifts= params['max_shifts'],strides = params['strides'],
                            overlaps=params['overlaps'], max_deviation_rigid=params['max_deviation_rigid'],
                            shifts_opencv =params['shifts_opencv'],nonneg_movie = True,
                            border_nan=params['border_nan'])
        
        mc.motion_correct(save_movie=True)
        registered = cm.load(mc.mmap_file)
        
        tif_name = os.path.join(self.tempfolder,tempname+'_registered'+'_.tif')
        io.imsave(tif_name,registered,plugin='tifffile')
        
        return mc
        
        # Runs above but on stack you give it