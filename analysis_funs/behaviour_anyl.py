# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:34:33 2025

@author: dowel
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils.utils_general import utils_general as ug
from rdp import rdp as rdp
from sklearn.decomposition import PCA
#%%
class ET_behaviour:
    def __init__(self,df):
        self.ft = df
        ft = self.ft
    
        # Get odour channel
        mfc = ft[['mfc1_stpt','mfc2_stpt','mfc3_stpt']].to_numpy()
        mfc_var = np.var(mfc,axis=0)
        used_mfcs = np.where(mfc_var>0)[0]
        t_mfcs = mfc[:,used_mfcs]
        mfc_mean = np.mean(t_mfcs,axis=0)
        odour = t_mfcs[:,np.argmin(mfc_mean)]
        ins = odour>0
        
        blockstart,blocksize = ug.find_blocks(ins,mergeblocks=True,merg_threshold =10)
        self.blockstart = blockstart
        self.blocksize = blocksize
        # Get plume size
        blockstart,blocksize = ug.find_blocks(ins)
        x = ft['ft_posx'].to_numpy()
        y = ft['ft_posy'].to_numpy()
        x = x-x[blockstart[0]]
        y = y-y[blockstart[0]]
        self.x = x
        self.y = y
        self.topcs = ug.load_pick('Y:\\Data\\Behaviour\\AndyVanilla\\constant_plume_AS\AndyDataReturnPCs_98.pkl')
    def eigen_return(self):
        """
        Sample return x,y trajectories onto same timebase and do PCA
        """
        x = self.x
        y = self.y
        ft = self.ft
        blockstart = self.blockstart
        blocksize = self.blocksize
        entries = blockstart[1:]
        
        exits = blockstart+blocksize-1
        
        ex_en = np.zeros((len(exits),2),dtype='int')-1
        ex_en[:,0] = exits
        ex_en[:len(entries),1] = entries
        tnew = np.arange(0,400)
        dlen = np.sum(ex_en[:,1]>0)
        data = np.zeros((dlen,800))
        for i,en in enumerate(ex_en):
            if en[1]==-1:
                break
            dx = np.arange(en[0],en[1])
            tx = x[dx]
            ty = y[dx]
            if np.min(tx)<0:
                tx = tx*-1
            ty = ty-ty[0]
            tx = tx-tx[0]
            ty = ty/np.max(np.abs(ty))
            tx = tx/np.max(np.abs(tx))
            told= np.linspace(0,399,len(dx))
            xi = np.interp(tnew,told,tx)
            yi = np.interp(tnew,told,ty)
            arr = np.append(xi,yi)
            data[i,:] =arr
            
        return data
    def extract_metrics(self):
        """
        Metrics: 
            Distance from plume x
            'Var in distance from plume'
            Entries per m x
            
            Plume velocity x
            Time to returns x
            Variance in time of returns x
            Number of RDP segments per return - Return regularity, perhaps. Ideal number is 2 - 1 inbound 1 outbound x
            Len last entry x
            Side fidelity x
            
            Extras:
                Time in plume
                Len y in plume
                
        """
        
        x = self.x
        y = self.y
        ft = self.ft
        blockstart = self.blockstart
        blocksize = self.blocksize
        px = x[blockstart]
        px = np.round(px)
        pwidth = np.abs(px[1])*2
        
        
        # Pair up entries and exits
        
        entries = blockstart[1:]
        
        exits = blockstart+blocksize-1
        
        ex_en = np.zeros((len(exits),2),dtype='int')-1
        ex_en[:,0] = exits
        ex_en[:len(entries),1] = entries # column 1 exits, column 2 entries, -1 in column 2 means animal did not return at end
        
        data = np.zeros(13)
        data[0] = self.get_max_distance(ft,ex_en,pwidth)
        data[1] = np.sqrt(self.var_of_max_distance(ft,ex_en,pwidth))
        data[2] = self.entries_per_m(ft,ex_en)
        data[3] = self.plume_velocity(ft,ex_en,blockstart)
        data[4] = self.time_to_returns(ft,ex_en)
        data[5] = np.sqrt(self.var_of_returns(ft,ex_en))
        data[6] = self.rdp_segments(ft,ex_en,avtype='mean')
        data[7] = y[blockstart[-1]]- y[blockstart[0]] # Length ascended up plume
        data[8] = self.side_fidelity(ft,ex_en)
        data[9:11] = self.return_mse(ft,ex_en) # Straightness of returns and exits
        data[11] = self.sum_an_return(ft,ex_en)
        data[12] = self.time_in_plume(ft,ex_en)
        # ed = self.eigen_return()
        # proj = np.matmul(ed,self.topcs.T)
        # projmean = np.mean(proj,axis=0)
        # data = data.append(data,projmean)
        return data
    
    def get_max_distance(self,ft,ex_en,pwidth,avtype='median'):
        x = self.x
        dlen = np.sum(ex_en[:,1]>0)
        data = np.zeros(dlen)
        for i,en in enumerate(ex_en):
            if en[1]==-1:
                break
            dx = np.arange(en[0],en[1])
            tx = np.abs(x[dx])-pwidth/2
            data[i] = np.max(tx)
        
        if avtype=='median':
            tdat = np.median(data)
        elif avtype=='mean':
            tdat = np.mean(data)
                
        return tdat
    def var_of_max_distance(self,ft,ex_en,pwidth):
        x = self.x
        dlen = np.sum(ex_en[:,1]>0)
        data = np.zeros(dlen)
        for i,en in enumerate(ex_en):
            if en[1]==-1:
                break
            dx = np.arange(en[0],en[1])
            tx = np.abs(x[dx])-pwidth/2
            data[i] = np.max(tx)
        
        
        tdat = np.var(data)
                
        return tdat
    
    def entries_per_m(self,ft,ex_en):
        y = self.y
        dlen = np.sum(ex_en[:,1]>0)
        ent_len  = y[ex_en[dlen,1]]-y[ex_en[0,1]]
        return dlen/ent_len
        
    def plume_velocity(self,ft,ex_en,blockstart):
        tt = ug.get_ft_time(ft)
        y = self.y
        st = blockstart[0]
        ed = ex_en[-1,0]
        dist = y[ed]-y[st] # difference between first plume on and last exit
        dt = tt[ed]-tt[st]
        return dist/dt
        
    def time_to_returns(self,ft,ex_en,avtype='median'):
        tt = ug.get_ft_time(ft)
        dlen = np.sum(ex_en[:,1]>0)
        data = np.zeros(dlen)
        for i,en in enumerate(ex_en):
            if en[1]==-1:
                break
            data[i] = tt[en[1]]-tt[en[0]]
            
        if avtype=='median':
            tdat = np.median(data)
        elif avtype=='mean':
            tdat = np.mean(data)
        
        return tdat
    def var_of_returns(self,ft,ex_en):
        tt = ug.get_ft_time(ft)
        dlen = np.sum(ex_en[:,1]>0)
        data = np.zeros(dlen)
        for i,en in enumerate(ex_en):
            if en[1]==-1:
                break
            data[i] = tt[en[1]]-tt[en[0]]
            
        tdat = np.var(data)
        
        return tdat
    def rdp_segments(self,ft,ex_en,avtype='median'):
        x = self.x
        y = self.y
        
        dlen = np.sum(ex_en[:,1]>0)
        data = np.zeros(dlen)
        arr = np.append(x[:,np.newaxis],y[:,np.newaxis],axis=1)
        for i,en in enumerate(ex_en):
            if en[1]==-1:
                break
            dx = np.arange(en[0],en[1])
            tarr = arr[dx,:]
            m = rdp(tarr,epsilon=5,return_mask=True)
            data[i] = np.sum(m)-1
        if avtype=='median':
            tdat = np.median(data)
        elif avtype=='mean':
            tdat = np.mean(data)
        
        return tdat
            
    def side_fidelity(self,ft,ex_en):
        x = self.x
        data = np.zeros(len(ex_en))
        for i,en in enumerate(ex_en):
            data[i] = np.sign(x[en[0]])
            
        smn = np.sign(np.sum(data))
        tdat = np.sum(data==smn)/len(data)
        if smn==0:
            tdat = 0.5
        return tdat
    
    def return_mse(self,ft,ex_en,avtype='median'):
        
        x = self.x
        y = self.y
        dlen = np.sum(ex_en[:,1]>0)
        data = np.zeros((dlen,2))
        for i,en in enumerate(ex_en):
            if en[1]==-1:
                break
            dx = np.arange(en[0],en[1])
            tx = x[dx]
            ty = y[dx]
            # Normalisation step to account for returns of different lengths, meaning we are just looking at shape
            tx = tx-tx[0]
            ty = ty-ty[0]
            
            tx = tx/np.max(np.abs(tx))
            ty = ty/np.max(np.abs(ty))
            dxm = np.argmax(np.abs(tx))
            
        
        
        
            xmax = tx[dxm]
            ymax = ty[dxm]
            # y = mx
            m1 = ymax/xmax
        
            ypred1 = tx[:dxm]*m1

            # y = mx +c
            c = ty[dxm]
            m2 = (ty[-1]-ymax)/(tx[-1]-xmax)
            if (tx[-1]-xmax)==0:
                print(len(tx))
            
            ypred2 = (tx[dxm:]-tx[dxm])*m2+c
            
            # plt.figure()
            # plt.plot(tx,ty,color='r')
            # plt.plot(tx[:dxm],ypred1,color='k')
            # plt.plot(tx[dxm:],ypred2,color='k')
            # mse_leave = np.mean((ypred1-ty[:dxm])**2)
            # print(mse_leave)
            
            data[i,0] = np.mean((ypred1-ty[:dxm])**2) # leave
            data[i,1] = np.mean((ypred2-ty[dxm:])**2) #return
            
        if avtype=='median':
            tdat = np.nanmedian(data,axis=0)
        elif avtype=='mean':
            tdat = np.nanmean(data,axis=0)  
        return tdat
    def sum_an_return(self,ft,ex_en,avtype='median'):
        
        heading = ft['ft_heading']
        dlen = np.sum(ex_en[:,1]>0)
        data = np.zeros(dlen)
        for i,en in enumerate(ex_en):
            if en[1]==-1:
                break
            dx = np.arange(en[0],en[1])
            th = heading[dx]
            th = np.unwrap(th)
            data[i] = th[-1]-th[0]
            
        if avtype=='median':
            tdat = np.nanmedian(data)
        elif avtype=='mean':
            tdat = np.nanmean(data)  
            
        
        return tdat
    def time_in_plume(self,ft,ex_en,avtype='median'):
        tt = ug.get_ft_time(ft)
        dshape = np.shape(ex_en)
        data = np.zeros(dshape[0]-1)
        for i in range(dshape[0]-1):
            
            data[i] = tt[ex_en[i+1,0]]-tt[ex_en[i,1]]
        if avtype=='median':
            tdat = np.nanmedian(data)
        elif avtype=='mean':
            tdat = np.nanmean(data)
        return tdat
    def plot_traj(self):
        x = self.x
        y = self.y
        
        
        plt.fill([-25,25,25,-25],[0,0,np.max(y),np.max(y)],color=[0.8,0.8,0.8])
        
        plt.plot(x,y,color='k')
        plt.gca().set_aspect('equal')
        plt.show()
        
        
        
        
        
        
        
        
    