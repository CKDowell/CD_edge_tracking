# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:57:07 2024

@author: dowel
"""

from analysis_funs.CX_imaging import CX
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
from analysis_funs.utilities import funcs as fn
import matplotlib as mpl
import pickle
from matplotlib import cm
from analysis_funs.regression import fci_regmodel
from Utilities.utils_plotting import uplt
from Utilities.utils_general import utils_general as ug
#%%
class CX_tan:
    def __init__(self,datadir,tnstring='0_fsbtn',Andy=False,span=500):
        
        
        d = datadir.split("\\")
        self.name = d[-3] + '_' + d[-2] + '_' + d[-1]
        if Andy==False:
            cx = CX(self.name,['fsbTN'],datadir)
            self.pv2, self.ft, self.ft2, self.ix = cx.load_postprocessing()
        else:
            post_processing_file = os.path.join(datadir,'postprocessing.h5')
            self.pv2 = pd.read_hdf(post_processing_file, 'pv2')
            self.ft2 = pd.read_hdf(post_processing_file, 'ft2')
            column_names = self.pv2.columns
            self.pv2.rename(columns={column_names[1]:tnstring+'_raw',column_names[2]:tnstring},inplace=True)
        print(self.pv2.columns)
        self.fc = fci_regmodel(self.pv2[[tnstring]].to_numpy().flatten(),self.ft2,self.pv2)
        
        #self.ca = self.fc.ca
        self.ca = self.pv2[[tnstring]].to_numpy() # keeps nan values, which are useful for plotting
        self.ca_no_nan = self.fc.ca
        self.fc.rebaseline(span=span,plotfig=False)
        self.ca_rebase = self.fc.ca
        self.tnstring = tnstring
        
    def reinit_fc(self):    
        self.fc = fci_regmodel(self.pv2[[self.tnstring]].to_numpy().flatten(),self.ft2,self.pv2)
        
    def get_jumps(self,time_threshold=60):
        # Function will find jump instances in the data and output the indices
        ft2 = self.ft2
        pv2 = self.pv2
        jumps = ft2['jump']
        ins = ft2['instrip']
        times = pv2['relative_time']
       
        insd = np.diff(ins)
        ents = np.where(insd>0)[0]+1
        exts = np.where(insd<0)[0]+1 
        jd = np.diff(jumps)
        jn = np.where(np.abs(jd)>0)[0]
        jkeep = np.where(np.diff(jn)>1)[0]
        jn = jn[jkeep]
        jns = np.sign(jd[jn])

        time_threshold = 60
        # Pick the most common side
        v,c = np.unique(jns,return_counts=True)
        side = v[np.argmax(c)]
        self.side = side # -1 is leftward jumps ie tracking right side, ie left goal
        # Get time of return: choose quick returns
        dt = []
        for i,j in enumerate(jn):
            ex = exts-j
            ie = np.argmin(np.abs(ex))
            t_ent = ie+1
            sub_dx = exts[ie]
            tdx = np.arange(ents[ie],ents[t_ent],step=1,dtype='int')
            dt.append(times[tdx[-1]]-times[sub_dx])
        this_j = jn[np.logical_and(jns==side, np.array(dt)<time_threshold)]
        
        out_dx = np.zeros((len(this_j),3),dtype='int')
        for i,j in enumerate(this_j):
            ex = exts-j
            ie = np.argmin(np.abs(ex))
            t_ent = ie+1
            sub_dx = exts[ie]
            ent = ents[ie]
            ent2 = ents[t_ent]
            out_dx[i,:] = np.array([ent,sub_dx,ent2],dtype='int')
        return out_dx
    
    def get_entries_exits(self,ent_duration=0.5): 
        #Funciton gets all entries and exits to the plume
        ins = self.ft2['instrip'].to_numpy()
        tt = self.pv2['relative_time'].to_numpy()
        td = np.mean(np.diff(tt))
        block,blocksize = ug.find_blocks(ins)
        thresh = np.round(td/ent_duration).astype(int)
        bdx = blocksize>=thresh
        entries = block[bdx]
        exits = entries+blocksize[bdx]
        return entries, exits
    def get_entries_exits_like_jumps(self,ent_duration=0.5,odour='ACV'):
        if odour =='ACV':
            ins = self.ft2['instrip'].to_numpy()
        elif odour=='Oct':
            ins = self.ft2['mfc3_stpt'].to_numpy()>0
        tt = self.pv2['relative_time'].to_numpy()
        td = np.mean(np.diff(tt))
        block,blocksize = ug.find_blocks(ins)
        block_o = block.copy()
        thresh = np.round(td/ent_duration).astype(int)
        bdx = blocksize>=thresh
        block = block[bdx]
        blocksize = blocksize[bdx]
        e_ex = np.zeros((len(block),3),dtype='int')
        for i,b in enumerate(block):
            e_ex[i,0] = b
            e_ex[i,1] = b+blocksize[i]
            try:
                next_block = np.min(block_o[block_o>b])
                e_ex[i,2] = next_block
            except:
                e_ex[i,2] = 0
        if e_ex[-1,2]==0: # clip off last epoch if animal does not return to plume. V common event
            e_ex = e_ex[:-1,:]
        return e_ex
        
        
    def mean_traj_nF(self,use_rebase = True,tnstring='0_fsbtn'):
        """
        Function outputs mean trajectory of animal entering and exiting the plume
        alongside the mean fluorescence

        Returns
        -------
        None.

        """
        plume_centres = [0,210,420]
        if use_rebase:
            ca = self.fc.ca
        else:
            ca = self.pv2[tnstring]
            
        ft2 = self.ft2
        pv2 = self.pv2
        ins = ft2['instrip']
        x = ft2['ft_posx'].to_numpy()
        y = ft2['ft_posy'].to_numpy()
        times = pv2['relative_time']
        x,y = self.fc.fictrac_repair(x,y)
        expst = np.where(ins==1)[0][0]
        x = x-x[expst]
        y = y-y[expst]
        insd = np.diff(ins)
        ents = np.where(insd>0)[0]+1
        exts = np.where(insd<0)[0]+1
        ents_O = ents.copy()
        exts_O = ents.copy()
        ents = ents[1:]
        exts = exts[1:]
        # Need to pick a side
        if len(ents)>len(exts):
            ents = ents[:-1]
        
        ent_x = np.round(x[ents])
        ex_x = np.round(x[exts])
        sides = np.zeros(len(ent_x))
        plume_centre = np.zeros(len(ent_x))
        
        
        # 1 -1 indicates sides of entry/exit
        # -0.9 0.9 indicates crossing over from left and right
        for i,x1 in enumerate(ent_x):
            x2 = ex_x[i]
            pcd = plume_centres-np.abs(x1)
            
            pi = np.argmin(np.abs(pcd))
            pc = plume_centres[pi]*np.sign(x1)
            plume_centre[i] = pc
            s_en = np.sign(x1-pc)
            s_ex = np.sign(x2-pc)
            if s_en==s_ex:
                sides[i] = s_en
            elif s_en!=s_ex:
               
                sides[i] = s_en+(s_ex/10)
                
                    
        v,c = np.unique(sides,return_counts=True)
        
        ts = v[np.argmax(c)]
        #print(ts)
        t_ents = ents[sides==ts]
        t_exts = exts[sides==ts]
        t_pc = plume_centre[sides==ts]
        # initialise arrays
        trajs = np.empty((100,2,len(t_ents)))
        Ca = np.empty((100,len(t_ents)))
        trajs[:] = np.nan
        Ca[:] = np.nan
        for i,en in enumerate(t_ents):
            prior_ex = exts_O-en
            prior_ex = prior_ex[prior_ex<0]
            pi = np.argmax(prior_ex)
            ex1 = exts_O[pi]
            dx1 = np.arange(ex1,en,dtype=int)
            dx2 = np.arange(en,t_exts[i],dtype=int)
            x1 = x[dx1]-t_pc[i]
           # print(t_pc[i])
            y1 = y[dx1]
            ca1 = ca[dx1]
            # print(x[dx2[0]]-t_pc[i])
            # print(x[dx2[-1]]-t_pc[i])
            # print(x[dx1[-1]]-t_pc[i])
            
            if np.sign(x[dx1[0]]-t_pc[i])!=np.sign(x[dx1[-1]]-t_pc[i]):
                continue
            # print('aaa')
            x2 = x[dx2]-t_pc[i]
            y2 = y[dx2]
            ca2 = ca[dx2]
            y2 = y2-y1[0]
            y1 = y1-y1[0]
            
            x1 = x1*ts
            x2 = x2*ts
            
            x1d = 5-x1[0]
            x1 = x1
            x2 = x2
            #Interpolate onto timebase: return
            old_time = dx1-dx1[0]
            new_time = np.linspace(0,max(old_time),50)
            x_int = np.interp(new_time,old_time,x1)
            y_int = np.interp(new_time,old_time,y1)
            ca_int = np.interp(new_time,old_time,ca1)
            trajs[:50,0,i] = x_int
            trajs[:50,1,i] = y_int
            Ca[:50,i] = ca_int
            
            #Interpolate onto timebase: in plume
            old_time = dx2-dx2[0]
            new_time = np.linspace(0,max(old_time),50)
            x_int = np.interp(new_time,old_time,x2)
            y_int = np.interp(new_time,old_time,y2)
            ca_int = np.interp(new_time,old_time,ca2)
            trajs[50:,0,i] = x_int
            trajs[50:,1,i] = y_int
            Ca[50:,i] = ca_int
            
        traj_mean = np.nanmean(trajs,axis=2)
        Ca_mean = np.nanmean(Ca,axis=1)
        return traj_mean,Ca_mean
    # def orientation_bin_returns(self,binwidth):
    #     jumps = self.get_jumps()
    #     ca = self.ca
    #     heading = self.ft2['ft_heading'].to_numpy()
    #     for j in jumps:
        
    def mean_traj_heat(self,xoffset=0,set_cmx =False,cmx=1):
        trj,ca = self.mean_traj_nF()
        colour = ca
        if set_cmx==False:
            cmx = np.max(np.abs(ca))
        c_map = plt.get_cmap('coolwarm')
        cnorm = mpl.colors.Normalize(vmin=-cmx, vmax=cmx)
        scalarMap = cm.ScalarMappable(cnorm, c_map)
        c_map_rgb = scalarMap.to_rgba(colour)
        yrange = np.array([min(trj[:,1]),max(trj[:,1])])
        plt.fill([-5+xoffset,5+xoffset,5+xoffset,-5+xoffset],yrange[[0,0,1,1]],color=[0.7,0.7,0.7])
        
        for i in range(len(ca)-1):
            x = trj[i:i+2,0]
            y = trj[i:i+2,1]
            #ca = np.mean(ca[i:i+2])
            plt.plot(x+xoffset,y,color=c_map_rgb[i,:])
            
            
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    def mean_ca_realtime_jump(self,t_length,norm=False):
        jumps = self.get_jumps()
        tbins = t_length*10
        ca = np.ones((tbins,len(jumps)))*1000
        for i,j in enumerate(jumps):
           # print(j)
            jend = j[2]
            jstart = j[1]
            jrange = np.arange(jstart,jend,dtype=int)
            if len(jrange)>tbins:
                
                #print('Too long', len(jrange), tbins)
                jrange = jrange[-tbins:]
                #print(jrange)
           
            tca = self.ca
            if norm:
                jn = np.mean(tca[jstart-5:jstart])
                jn2 = np.mean(tca[jend-5:jend])
                print(jn)
                ca[-len(jrange):,i] = tca[jrange]-jn2
            else:
                
                ca[-len(jrange):,i] = tca[jrange].ravel()
            
        #phases = np.flipud(phases)
        ca[ca==1000] = np.nan
        return ca
    
    def mean_traj_jump(self,timethreshold=60,bins=100):
        jumps = self.get_jumps()
        newtime = np.linspace(0,1,bins)
        jseries = self.ft2['jump'].to_numpy()
        retxy = np.zeros((bins,2,len(jumps)))
        levxy = np.zeros((bins,2,len(jumps)))
        retact = np.zeros((bins,len(jumps)))
        levact = np.zeros((bins,len(jumps)))
        ins = self.ft2['instrip']
        inst = np.where(ins)[0][0]
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        x,y = self.fc.fictrac_repair(x,y)
        x = x-x[inst]
        y = y-y[inst]
        x = -x*self.side
        ca = self.ca_rebase
        for i,j in enumerate(jumps):
            bdx = np.arange(j[0],j[1])
            adx = np.arange(j[1],j[2])
            tj = jseries[j[2]]
            
            tx = x-x[j[1]-1]
            ty = y-y[j[1]-1]
            
            rx = tx[adx]
            ry = ty[adx]
            rca = ca[adx]
            
            lx = tx[bdx]
            ly = ty[bdx]
            lca = ca[bdx]
            
            
            
            oldtime = np.linspace(0,1,len(rx))
            retxy[:,0,i] = np.interp(newtime,oldtime,rx)
            retxy[:,1,i] = np.interp(newtime,oldtime,ry)
            retact[:,i] = np.interp(newtime,oldtime,rca)
            
            
            oldtime = np.linspace(0,1,len(lx))
            levxy[:,0,i] = np.interp(newtime,oldtime,lx)
            levxy[:,1,i] = np.interp(newtime,oldtime,ly)
            levact[:,i] = np.interp(newtime,oldtime,lca)      
        act = np.append(levact,retact,axis=0)
        trajs = np.append(levxy,retxy,axis=0)
        return act,trajs
    def plot_mean_traj_jump(self,timethreshold=60,bins=100,cmin=-1,cmax=1):
        act,trajs = self.mean_traj_jump(timethreshold=60,bins=100)
        am = np.mean(act,axis=1)
        xy = np.mean(trajs,axis=2)
        am[am<cmin] = cmin
        am[am>cmax] = cmax
        plt.figure()
        ymin = np.min(xy[:,1])
        ymax = np.max(xy[:,1])
        xarray = np.array([-10,0,0,-10])
        yarray = np.array([ymin,ymin,0,0])
        plt.fill(xarray,yarray,color=[0.7,0.7,0.7])
        yarray = np.array([0,0,ymax,ymax])
        plt.fill(xarray-3,yarray,color=[0.7,0.7,0.7])
        ax = plt.gca()
        uplt.coloured_line_simple(xy[:,0],xy[:,1],am,'coolwarm',cmin,cmax)
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        
       #uplt.coloured_line(xy[:,0], xy[:,1], am, ax,cmap='coolwarm')
        
    def get_jumps(self,time_threshold=60):
        # Function will find jump instances in the data and output the indices
        ft2 = self.ft2
        pv2 = self.pv2
        jumps = ft2['jump']
        ins = ft2['instrip']
        times = pv2['relative_time']
       
        insd = np.diff(ins)
        ents = np.where(insd>0)[0]+1
        exts = np.where(insd<0)[0]+1 
        jd = np.diff(jumps)
        jn = np.where(np.abs(jd)>0)[0]
        jkeep = np.where(np.diff(jn)>1)[0]
        jn = jn[jkeep]
        jns = np.sign(jd[jn])

        time_threshold = 60
        # Pick the most common side
        v,c = np.unique(jns,return_counts=True)
        side = v[np.argmax(c)]
        self.side = side
        # Get time of return: choose quick returns
        dt = []
        for i,j in enumerate(jn):
            ex = exts-j
            ie = np.argmin(np.abs(ex))
            t_ent = ie+1
            sub_dx = exts[ie]
            tdx = np.arange(ents[ie],ents[t_ent],step=1,dtype='int')
            dt.append(times[tdx[-1]]-times[sub_dx])
        this_j = jn[np.logical_and(jns==side, np.array(dt)<time_threshold)]
        
        out_dx = np.zeros((len(this_j),3),dtype='int')
        for i,j in enumerate(this_j):
            ex = exts-j
            ie = np.argmin(np.abs(ex))
            t_ent = ie+1
            sub_dx = exts[ie]
            ent = ents[ie]
            ent2 = ents[t_ent]
            out_dx[i,:] = np.array([ent,sub_dx,ent2],dtype='int')
        return out_dx
        