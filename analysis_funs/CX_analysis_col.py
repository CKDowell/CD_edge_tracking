# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:35:43 2024

@author: dowel
"""
#%%
import src.utilities.funcs as fc
from analysis_funs.CX_imaging import CX
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
from analysis_funs.utilities import funcs as fn
import pickle
from scipy.stats import circmean, circstd
from Utilities.utils_general import utils_general as ug
from statistics import mode
#%%
class CX_a:
    def __init__(self,datadir,regions =['eb','fsb'],Andy=False,denovo=True,yoking=True,stim=False,suspend=False):
        # Will need to edit more if yoking to PB and multiple FSB layers
        self.stab = regions[0]
        if self.stab=='pb':
            # Array to convert numbering of PB glomeruli from logical space to anatomical space
            self.logic2anat = np.array([8,0,9,1,10,2,11,3,12,4,13,5,14,6,15,7],dtype='int')
        self.datadir = datadir
        d = datadir.split("\\")
        name = d[-3] + '_' + d[-2] + '_' + d[-1]
        self.name = name
        if Andy=='hDeltaC':
            savepath = os.path.join(datadir,"all_info_eb_fb.csv")
            data = pd.read_csv(savepath)
            self.alldat = data
            
            w_eb_o = data['wedges_eb']
            w_fb_u_o = data['wedges_fb_upper']
            w_fb_l_o = data['wedges_fb_lower']

            # Repair string data
            for i,e in enumerate(w_eb_o):
                enum_str = e.replace('[','').replace(']','').replace('\n','').split()
                enum = [float(num) for num in enum_str]
                if i==0:
                    w_eb = np.empty([len(w_eb_o), len(enum)])
                w_eb[i,:] =enum

            for i,e in enumerate(w_fb_u_o):
                enum_str = e.replace('[','').replace(']','').replace('\n','').split()
                enum = [float(num) for num in enum_str]
                if i==0:
                    w_fb_u = np.empty([len(w_fb_u_o), len(enum)])
                w_fb_u[i,:] =enum    

            for i,e in enumerate(w_fb_l_o):
                enum_str = e.replace('[','').replace(']','').replace('\n','').split()
                enum = [float(num) for num in enum_str]
                if i==0:
                    w_fb_l = np.empty([len(w_fb_l_o), len(enum)])
                w_fb_l[i,:] =enum 
            
            
            
            
            self.pdat = {'offset_eb_phase': data['offset_phase_eb'].to_numpy(),
                         'offset_fsb_upper_phase': data['offset_phase_fb_upper'].to_numpy(),
                         'offset_fsb_lower_phase': data['offset_phase_fb_lower'].to_numpy(),
                         'wedges_fsb_lower':w_eb,
                         'wedges_fsb_upper': w_fb_u,
                         'wedges_fsb_lower': w_fb_l}
            self.amp = data['fitted_amplitude_fb_upper']
            self.amp_lower = data['fitted_amplitude_fb_lower']
            self.amp_eb = data['fitted_amplitude_eb']
                        
            self.ft2 = data[['x','y','seconds','heading','instrip']]
            self.ft2 = self.ft2.rename(columns={'x': 'ft_posx', 'y': 'ft_posy',
                                              'seconds':'relative_time','heading': 'ft_heading'})
            x = self.ft2['instrip'].to_numpy()
            x[x=='False'] = '0.0'
            x = np.array(x[:],'float')
            self.ft2['instrip'] = x
            
            self.pv2 = self.ft2[['relative_time','ft_posx']]
        elif Andy =='hDeltaB':
            # Rename fsb columns
            self.cx = CX(name,regions,datadir)
            self.pv2, self.ft, self.ft2, ix = self.cx.load_postprocessing()        
            self.phase,self.phase_offset,self.amp = self.cx.unyoked_phase('fb')
            self.phase_eb,self.phase_offset_eb,self.amp_eb = self.cx.unyoked_phase(self.stab)
            self.pdat = self.cx.phase_yoke(self.stab,['fb'],self.ft2,self.pv2)
            for i in range(16):
                self.pv2 = self.pv2.rename(columns={str(i) +'_fb': str(i)+'_fsb'})
            self.pdat['fit_wedges_fsb'] = self.pdat.pop('fit_wedges_fb')
            self.pdat['all_params_fsb'] = self.pdat.pop('all_params_fb')
            self.pdat['amp_fsb'] = self.pdat.pop('amp_fb')
            self.pdat['offset_fsb_phase'] = self.pdat.pop('offset_fb_phase')
        else: 
            self.cx = CX(name,regions,datadir)
            self.pv2, self.ft, self.ft2, ix = self.cx.load_postprocessing()  
            self.pv2 = self.pv2.drop(columns=self.pv2.filter(regex='fsbtn').columns) # needed to drop any whole TN masks
            if stim:
                self.interpolate_over_stim(regions) # interpolates signal over shutter blockage with stimulation
            #self.pv2 = self.pv2.drop(columns=['0_fsbtn']) # drop any reference to tangential neurons
            if suspend:
                self.suspend_heading_correction()
            
            x= self.ft2['ft_posx']
            y = self.ft2['ft_posy']
            heading = self.ft2['ft_heading']
            if set(['bump']).issubset(self.ft2):
                self.ft2['bump'][np.isnan(self.ft2['bump'])] = 0
                if np.sum(np.abs(self.ft2['bump']))>0:
                    x,y,heading = self.bumpstraighten(x.to_numpy(),y.to_numpy(),heading)    
                    self.ft2_original = self.ft2.copy()
                    self.ft2['ft_posx'] = x
                    self.ft2['ft_posy'] = y
                    self.ft2['ft_heading'] = heading
            
            if denovo:
                
                    
                if yoking:
                    # Correct for bumps

                    self.phase,self.phase_offset,self.amp = self.cx.unyoked_phase(regions[1])
                    self.phase_offset = self.phase_offset.to_numpy()
                    self.phase = self.phase.reshape(-1, 1)
                    self.phase_offset = self.phase_offset.reshape(-1,1)
                    self.amp = self.amp.reshape(-1,1)
                    if len(regions)>2: 
                        for i,r in enumerate(regions[2:]):
                            p,o,a = self.cx.unyoked_phase(r)
                            o = o.to_numpy()
                            p = p.reshape(-1,1)
                            o = o.reshape(-1,1)
                            a = a.reshape(-1,1)
                            self.phase = np.append(self.phase,p,axis =1)
                            self.phase_offset = np.append(self.phase_offset,o,axis =1)
                            self.amp = np.append(self.amp,a,axis = 1)
                    self.phase_eb,self.phase_offset_eb,self.amp_eb = self.cx.unyoked_phase(self.stab)
                    self.pdat = self.cx.phase_yoke(self.stab,regions[1:],self.ft2,self.pv2)
                else:
                    for i, r in enumerate(regions):
                        p,o,a = self.cx.unyoked_phase(r)
                        wedges = self.pv2.filter(regex=r)
                        wedges.fillna(method='ffill', inplace=True)
                        wedges = wedges.to_numpy()
                        
                        if i==0:
                            self.phase =p
                            self.amp = p
                            #self.phase_offset = o
                            #self.phase_offset = self.phase_offset.to_numpy()
                            self.phase = self.phase.reshape(-1, 1)
                            #self.phase_offset = self.phase_offset.reshape(-1,1)
                            self.amp = self.amp.reshape(-1,1)
                            self.pdat = {'wedges_' + r:wedges,'phase_'+r:p,'amp_'+r:a}
                            
                        else:
                            p = p.reshape(-1,1)
                            #o = o.reshape(-1,1)
                            a = a.reshape(-1,1)
                            self.phase = np.append(self.phase,p,axis =1)
                            #self.phase_offset = np.append(self.phase_offset,o,axis =1)
                            self.amp = np.append(self.amp,a,axis = 1)
                            self.pdat.update({'wedges_' + r:wedges,'phase_'+r:p,'amp_'+r:a})
                        
                        
                        
            else:
                loaddir = os.path.join(self.datadir,'processed','phase_dict.pkl')
                with open(loaddir, 'rb') as f:
                    self.pdat = pickle.load(f)
                self.phase_eb = self.pdat['phase_'+self.stab]
                self.amp_eb = self.pdat['amp_' + self.stab]
                if len(regions)>1:
                    self.phase = self.pdat['phase_'+regions[1]]
                    self.amp = self.pdat['amp_' + regions[1]]
                    self.phase = self.phase.reshape(-1, 1)
                    self.amp = self.amp.reshape(-1,1)
                    for i,r in enumerate(regions[2:]):
                        p = self.pdat['phase_'+r]
                        a = self.pdat['amp_'+r]
                        p = p.reshape(-1,1)                 
                        a = a.reshape(-1,1) 
                        self.phase = np.append(self.phase,p,axis =1)
                        self.amp = np.append(self.amp,a,axis = 1)
                else:
                    self.phase = self.pdat['phase_'+regions[0]]
                    self.amp = self.pdat['amp_' + regions[0]]
                    self.phase = self.phase.reshape(-1, 1)
                    self.amp = self.amp.reshape(-1,1)
            
    def save_phases(self):
        savedir = os.path.join(self.datadir,'processed','phase_dict.pkl')
        save_dict = self.pdat
        with open(savedir, 'wb') as f:
            pickle.dump(save_dict, f)
            
    def suspend_heading_correction(self):
        ft2 = self.ft2.copy()
        self.ft2_presuspend = ft2.copy()
        fx = ft2['fix_heading'].to_numpy()
        th = ft2['train_heading'].to_numpy()
        x = ft2['ft_posx'].to_numpy()
        y = ft2['ft_posy'].to_numpy()
        x,y = self.fictrac_repair(x,y)
        dx = np.diff(x.copy())
        dy = np.diff(y.copy())
        speed = np.sqrt(dx**2+dy**2)
        speed = np.append(0,speed)
        speed[np.abs(speed)>5]=0
        fx[fx==False] = 0
        fx[fx==True] = 1
        fx = np.array(fx, dtype=float)
        fx[np.isnan(fx)] = 0
        fx[fx>0] =1
        blocks,blocksize = ug.find_blocks(fx)
        th = fc.wrap(th,0,2*np.pi)
        th2 = th.copy()
        th[th2>np.pi] = -(2*np.pi-th2[th2>np.pi])
        plt.plot(x,y,linewidth=4)
        for i,b in enumerate(blocks):
            bdx = np.arange(b,b+blocksize[i])
            t_ang = th[bdx]
            ta = mode(t_ang)
            ft2.loc[bdx,'ft_heading'] = ta
            
            tdx = speed[bdx]*np.cos(ta)
            tdy = speed[bdx]*np.sin(ta)
            tx = np.cumsum(tdx)
            ty = np.cumsum(tdy)
            print(tx)
            
            #x[bdx] = tx+x[b-1]
            #x[bdx[-1]:] = x[bdx[-1]:]+x[bdx[-1]-1]
            print(x[bdx])
            #y[bdx] = ty+y[b-1]
            #y[bdx[-1]:] = y[bdx[-1]:]+y[bdx[-1]-1]
        plt.plot(x,y)  
        self.ft2 = ft2
        
        
    def interpolate_over_stim(self,rois):
        # Interpolates over LED stimulation artefacts
        from scipy.signal import find_peaks
        
        try :
            pv2 = self.pv2_original
        except:
            pv2 = self.pv2
        ft2 = self.ft2
        self.pv2_original = pv2.copy()
        led1 = (ft2['led1_stpt'].to_numpy()-1)*-1
        
        led1[led1>0] =1
        if np.sum(led1)==0:
            return
        bi,bs = ug.find_blocks(led1,mergeblocks=True,merg_threshold = 5)
        
        
        cnames = pv2.columns.to_list()
        
        for ir, r in enumerate(rois):
            roidx = [r in c for c in cnames ]
            y_all = pv2.iloc[:,roidx].to_numpy()
            ynew = y_all.copy()
                   
            t_all = np.arange(0,len(y_all)/10,.1)
            for iy in range(np.shape(y_all)[1]):
                y = y_all[:,iy]
                ynew[np.isnan(ynew[:,iy]),iy] = np.nanmean(y)
                y[np.isnan(y)] = 0
                
                for i in range(len(bs)):
                    seg = np.arange(bi[i]-5,bi[i]+bs[i]+5)
                    ty = y[seg]
                    ty = ty-np.percentile(ty,30)
                    ty[ty<0] = 0
                    tyo = y[seg]
                    tt = t_all[seg]
                    yp = np.percentile(ty,60)-np.percentile(ty,40)
                    peaks = find_peaks(ty,prominence=yp )[0]
                    
                    # Uncomment below for additional peaks if you think it is missing some 
                    
                    # pdiff = np.diff(peaks)
                    # mdiff = stats.mode(pdiff)
                    # pdub = (pdiff-mdiff[0])/mdiff[0]
                    # xtra = np.where(pdub>0.8)[0]
                    # for x in xtra:
                    #     pr = np.round(pdub[x]).astype('int')
                    #     for p in range(pr):
                      
                    #         peaks = np.append(peaks,peaks[x]+mdiff[0]*p)
                    
                    
                    peaks = np.append(0,peaks)
                    peaks = np.append(peaks,len(ty)-1)
                    peaks = np.unique(peaks)
                      
                    ny = np.interp(tt,tt[peaks],tyo[peaks])
                    ynew[seg,iy] = ny
                         
            yp = np.percentile(ynew,5,axis=0)
            pv2.iloc[:,roidx] = ynew-yp
        self.pv2 = pv2
    def von_mises_fit(self,gsigma=7,samsize=1000,regions=['fsb_upper']):
        """Function fits a von mises distribution to bump data
        This can then be used to fit a probability of pointing to plume edge"""
        from scipy import ndimage
        from scipy import stats
        bin_edges = np.linspace(-np.pi,np.pi,17)
        for i,r in enumerate(regions):
            bump = self.pdat['fit_wedges_' +r]
            if gsigma>0:
                bumpf = ndimage.gaussian_filter1d(bump,gsigma,axis=0)
            else:
                bumpf = bump
                
            #bumpf = bumpf-np.min(bumpf,axis=1)
            #bumpf = bumpf-np.sum(bumpf,axis=1)
            params = np.zeros((len(bumpf),3))
            predside = np.zeros((len(bumpf),2))
            for m,b in enumerate(bumpf):
                bm = bumpf[m,:]-min(bumpf[m,:])
                bm = bm/np.sum(bm)
                if np.mod(m,1000)==0:
                    print(m,' of ',len(bumpf))
                pdf = stats.rv_histogram((bm, bin_edges))
                samples = pdf.rvs(size=samsize)
                params[m,:] = stats.vonmises.fit(samples)
                
                # Right side tracker 
                pcd = stats.vonmises.cdf([-3*np.pi/4,-np.pi/4],params[m,0],params[m,1])
                pcdf = pcd[1]-pcd[0]
                predside[m,0] = pcdf
                
                # Left side tracker
                pcd = stats.vonmises.cdf([np.pi/4,3*np.pi/4],params[m,0],params[m,1])
                pcdf = pcd[1]-pcd[0]
                predside[m,1] = pcdf
            self.von_mises = {'params_'+r: params,
                              'predside_'+r:predside}
        
        
    def simple_raw_plot(self,plotphase=False,regions = ['fsb'],yeseb = True,yk='eb'):
        plt.figure(figsize=(5,10))
        phase = self.phase.copy() 
        ebs = []
        if yeseb:
            phase_eb = self.phase_eb.copy() 
            for i in range(16):
                ebs.append(str(i) +'_'+yk)
                
        for r in regions:
            for i in range(16):
                ebs.append(str(i) +'_' + r)
        
        eb = self.pv2[ebs].to_numpy()
        eb[np.isnan(eb)] = 0
        eb = eb/np.max(eb,axis=0)
       # eb[:,16:] = -eb[:,16:] +np.tile(np.max(eb[:,16:],axis=1)[:,np.newaxis],(1,16))
        print(np.shape(eb))
        t = np.arange(0,len(eb))
        plt.imshow(eb, interpolation='None',aspect='auto',cmap='Blues',vmax=np.nanpercentile(eb[:],99),vmin=np.nanpercentile(eb[:],5))
        if yeseb:
            new_phase = np.interp(phase_eb, (-np.pi, np.pi), (-0.5, 15.5))
            if plotphase:
                plt.plot(new_phase,t,color='r',linewidth=0.5)
            plt.plot([15.5,15.5],[min(t), max(t)],color='w')
            plt.xticks([0, 7, 15, 16,23, 31,32,40,48],
                       labels=['eb:1', 'eb:8', 'eb:16','fsb:1','fsb:8','fsb16','-$\pi$','0','$\pi$'],rotation=45)
            off = 0.5
        else:
            off = -15.5
        reps = np.shape(phase)[1]
        for i in range(reps):
            print(i)
            off = off+15
            o2 = off+16
            new_phase = np.interp(phase[:,i], (-np.pi, np.pi), (off, o2))
            if plotphase:
                plt.plot(new_phase,t,color='r',linewidth=0.5)
            off = off+1
        off = off+1
        new_heading = self.ft2['ft_heading'].to_numpy()
        new_heading = np.interp(new_heading, (new_heading.min(), new_heading.max()), (off+15, off+31))
        for i in range(reps):
            try:
                p = self.pdat['offset_'+ regions[i]+'_phase']
            except:
                p = self.pdat['phase_'+ regions[i]]
            #p = phase[:,i]
            new_phase = np.interp(p, (-np.pi, np.pi), (off+15, off+31))
            if plotphase:
                plt.plot(new_phase,t,color='b',linewidth=0.5)
        plt.plot(new_heading,t,color='k')
        sdiff = np.diff(self.ft2['instrip'].to_numpy())
        son = np.where(sdiff>0)[0]+1 
        soff = np.where(sdiff<0)[0]+1 
        for i,s in enumerate(son):
            plt.plot([off+15, off+31],[s, s],color=[1,0.5,0.5])
            plt.plot([off+15, off+31],[soff[i], soff[i]],color=[1,0.5,0.5])
            plt.plot([off+15,off+15],[s,soff[i]],color=[1,0.5,0.5])
            plt.plot([off+31,off+31],[s,soff[i]],color=[1,0.5,0.5])
            
            
        frate = np.mean(np.diff(self.pv2['relative_time']))
        yt = np.arange(0,max(t),60/frate)
        
        
        plt.yticks(yt,labels=yt*frate)
        plt.ylabel('Time (s)')
        plt.show()
    def entry_exit_phase(self):
        mult = float(180)/np.pi
        eb_phase = self.pdat['offset_eb_phase']*mult
        fsb_phase = self.pdat['offset_fsb_phase']*mult
        strip = self.ft2['instrip'].to_numpy()
        sdiff = np.diff(strip)
        son = np.where(sdiff>0)[0]+1
        soff = np.where(sdiff<0)[0]+1
        
        for i,s in enumerate(son):
            plt.figure()
            pdx = np.arange(s-1,soff[i],dtype='int')
            bdx = np.arange(s-50,s,dtype='int')
            #print(bdx)
            
            plt.scatter(eb_phase[bdx],fsb_phase[bdx],color='k',s=10)
            plt.scatter(eb_phase[pdx],fsb_phase[pdx],color='r',s=10)
            plt.plot(eb_phase[bdx],fsb_phase[bdx],color='k')
            plt.plot(eb_phase[pdx],fsb_phase[pdx],color='k')
            plt.xlim([-180,180])
            plt.ylim([-180,180])
            plt.plot([-180,180],[-180,180],color='k',linestyle='--')
            plt.xticks([-180,-90,0,90,180])
            plt.yticks([-180,-90,0,90,180])
            plt.ylabel('FSB phase')
            plt.xlabel('EB phase')
            plt.show()
    def simple_traj(self):
        plt.figure()
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        strip = self.ft2['instrip'].to_numpy()
        plt.plot(x,y,color='k')
        plt.scatter(x[strip>0],y[strip>0],c='r',s=15)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    def mean_phase_trans(self,tbef=5,taf=5,give_data=False,phase=[False]):
        if phase[0]==False:
            
            fsb_phase = self.pdat['offset_fsb_phase']
        else :
            fsb_phase = phase
        eb_phase = self.pdat['offset_eb_phase']
        strip = self.ft2['instrip'].to_numpy()
        sdiff = np.diff(strip)
        ts = self.pv2['relative_time'].to_numpy()
        tinc = np.mean(np.diff(ts))
        son = np.where(sdiff>0)[0]+1
        idx_bef = int(np.round(float(tbef)/tinc))
        idx_af = int(np.round(float(taf)/tinc))
        son = son[1:]
        mn_mat = np.zeros((len(son),idx_bef+idx_af+1))
        
        for i,s in enumerate(son):
            idx_array = np.arange(s-idx_bef-1,s+idx_af,dtype= int)
            if idx_array[-1]> len(eb_phase):
                nsum = np.sum(idx_array>len(eb_phase))
                idx_array = idx_array[idx_array<len(fsb_phase)]
                mn_mat[i,:-(nsum+1)] = eb_phase[idx_array]
            else:
                mn_mat[i,:] = eb_phase[idx_array]
        plt_mn = circmean(mn_mat,axis=0,high=np.pi,low=-np.pi)
        plt_mn_eb = plt_mn
        std = circstd(mn_mat,axis=0,high=np.pi,low=-np.pi)
        t = np.linspace(-tbef,taf,idx_bef+idx_af+1)
        t_eb =t.copy()
        
        plt.figure()
        fx = np.append(plt_mn+std,np.flipud(plt_mn-std))
        tx =  np.append(t,np.flipud(t))
        
        
        plt.fill(fx,tx,color = [0.6, 0.6, 0.6],zorder=0,alpha = 0.3)
        plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
        plt.plot(plt_mn,t,color='k',zorder=1)
        
        mn_mat2 = np.zeros((len(son),idx_bef+idx_af+1))
        for i,s in enumerate(son):
            idx_array = np.arange(s-idx_bef-1,s+idx_af,dtype= int)
            if idx_array[-1]> len(fsb_phase):
                nsum = np.sum(idx_array>len(fsb_phase))
                idx_array = idx_array[idx_array<len(fsb_phase)]
                mn_mat2[i,:-(nsum+1)] = fsb_phase[idx_array]
            else:
                mn_mat2[i,:] = fsb_phase[idx_array]
                
        plt_mn = circmean(mn_mat2,axis=0,high=np.pi,low=-np.pi)
        std = circstd(mn_mat2,axis=0,high=np.pi,low=-np.pi)
        t = np.linspace(-tbef,taf,idx_bef+idx_af+1)
        tx =  np.append(t,np.flipud(t))
        fx = np.append(plt_mn+std,np.flipud(plt_mn-std))
        plt.fill(fx,tx,color = [0.6, 0.6, 1],zorder=3,alpha = 0.3)
        plt.plot(plt_mn,t,color=[0.3,0.3,0.8],zorder=4)
        plt.plot([0,0],[-tbef,taf],color='k',linestyle='--',zorder=5)
        mn = -np.pi
        mx = np.pi
        plt.plot([mn,mx],[0,0],color='k',linestyle='--')
        plt.ylabel('Time (s)')
        plt.xlabel('Phase')
        plt.xticks([mn,mn/2,0,mx/2,mx],labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
        plt.xlim([mn,mx])
        plt.ylim([min(t), max(t)])
        plt.show()
        if give_data:
            # Interpolate onto a 5 Hz timebase
            t_norm = np.arange(-tbef,taf,0.2)
            plt_mn = np.interp(t_norm,t,plt_mn)
            plt_mn_eb = np.interp(t_norm,t_eb,plt_mn_eb)
            return plt_mn,plt_mn_eb,t_norm
            
        
        
        plt.figure()
        plt.plot(np.transpose(mn_mat),t,color='k',alpha=0.5)
        plt.plot([mn,mx],[0,0],color='k',linestyle='--')
        plt.show()
        plt.figure()
        plt.plot(np.transpose(mn_mat2),t,color='b',alpha=0.5)
        plt.plot([mn,mx],[0,0],color='k',linestyle='--')
        plt.show()
    def all_phase_arrow(self,tbef=3,taf=3,phase='offset_fsb_phase'):
        eb_phase = self.pdat['offset_eb_phase']
        fsb_phase = self.pdat[phase]
        #eb_phase = self.phase_eb
        #fsb_phase = self.phase
        mult = 3
        strip = self.ft2['instrip'].to_numpy()
        sdiff = np.diff(strip)
        ts = self.pv2['relative_time'].to_numpy()
        tinc = np.mean(np.diff(ts))
        
        son = np.where(sdiff>0)[0]+1
       
        soff = np.where(sdiff<0)[0]+1
        idx_bef = int(np.round(float(tbef)/tinc))
        idx_af = int(np.round(float(taf)/tinc))
        offset =0
        for i,s in enumerate(son):
            #plt.figure()
            if np.mod(i,10)==0:
                offset = 0
                plt.figure()
            so = soff[i]+idx_af
            print(i)
            s_on = son[i]-idx_bef
            sdx = np.arange(s_on,so,1)
            t = sdx-sdx[0]
            t = t*tinc
            t =t-tbef
            for v,vi in enumerate(sdx):
                if np.mod(v,1)==0:
                    m = eb_phase[vi]
                    x = mult*np.sin(m)
                    y = mult*np.cos(m)
                    plt.arrow(offset,t[v],x,y,color =[0,0,0])
                
                    m = fsb_phase[vi]
                    x = mult*np.sin(m)
                    y = mult*np.cos(m)
                    plt.arrow(offset,t[v],x,y,color =[0.6,0.6,1])
            plt.plot([-2+offset, 2+offset],[0,0],linestyle='--',color='k')
            plt.plot([-2+offset,2+offset],[t[-idx_af],t[-idx_af]],linestyle='--',color='k')
            offset = offset+mult*2
    def mean_jump_lines(self,fsb_names=['fsb_upper','fsb_lower'],p_amp=[False],time_threshold = 60):
         
        plt.figure(figsize=(20,20))
        from scipy.stats import circmean, circstd
        ft2 = self.ft2
        pv2 = self.pv2
        jumps = ft2['jump']
        ins = ft2['instrip']
        x = ft2['ft_posx'].to_numpy()
        y = ft2['ft_posy'].to_numpy()
        if p_amp[0]==False:
            p_amp =np.ones_like(x)
            print('blah')
        times = pv2['relative_time']
        x,y = self.fictrac_repair(x,y)
       
        insd = np.diff(ins)
        ents = np.where(insd>0)[0]+1
        exts = np.where(insd<0)[0]+1 
        jd = np.diff(jumps)
        jn = np.where(np.abs(jd)>0)[0]
        jkeep = np.where(np.diff(jn)>1)[0]
        jn = jn[jkeep]
        jns = np.sign(jd[jn])

        #time_threshold = 60
        # Pick the most common side
        v,c = np.unique(jns,return_counts=True)
        side = v[np.argmax(c)]
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


        side_mult = side*-1
        #Iterate
        x = x*side_mult
        offset = 0
        offsety = 0
        for i,j in enumerate(this_j):
            ex = exts-j
            ie = np.argmin(np.abs(ex))
            t_ent = ie+1
            sub_dx = exts[ie]
            phase = self.pdat['offset_eb_phase'].to_numpy()
            phase = phase.reshape(-1,1)
            for f in fsb_names:
                phase = np.append(phase,self.pdat['offset_' +f+ '_phase'].to_numpy().reshape(-1,1),axis=1)
            heading = self.ft2['ft_heading'].to_numpy()*side_mult
            phase = phase*side_mult
            amp = self.pdat['amp_eb']
            amp = amp.reshape(-1,1)
            for f in fsb_names:
                amp = np.append(amp,self.pdat['amp_'+f].reshape(-1,1),axis=1)
            # in plume
            ipdx = np.arange(ents[ie],ents[t_ent],step=1,dtype=int)
            
            t_p = phase[ipdx,1]
            ta = p_amp[ipdx]
            ta = ta-np.percentile(ta,5)
            t_t = self.pv2['relative_time'].to_numpy()
            t_t = t_t[ipdx]
            t_h = heading[ipdx]
            t_t = t_t-t_t[-1]            
            t_ext = sub_dx-ents[ie]
            #plt.plot(t_t+offset,t_p+offsety,color=[0.2,0.2,1])
            
            plt.scatter(t_t+offset,t_p+offsety,color=[0.2,0.2,1],s=5,zorder=10)
            plt.plot([t_t[0]+offset,t_t[-1]+offset],[0+offsety,0+offsety],color='k',linestyle='--')
            plt.plot([t_t[0]+offset,t_t[t_ext]+offset],[np.pi/2+offsety,np.pi/2+offsety],color=[0.8,0.2,0.2],linestyle='--')
            plt.plot([t_t[t_ext]+offset,t_t[-1]+offset],[-np.pi/2+offsety,-np.pi/2+offsety],color=[0.8,0.2,0.2],linestyle='--')
            plt.plot(t_t+offset,t_h+offsety,color='k')
            plt.fill(np.array([t_t[0],t_t[-1],t_t[-1],t_t[0],t_t[0]])+offset,np.array([-np.pi,-np.pi,np.pi,np.pi,-np.pi])+offsety,color=[0.9,0.9,0.9])
            plt.fill(np.array([t_t[0],t_t[t_ext],t_t[t_ext],t_t[0],t_t[0]])+offset,np.array([-np.pi,-np.pi,np.pi,np.pi,-np.pi])+offsety,color=[0.7,0.7,0.7])
            
            #plt.plot(t_t+offset,5*ta+offsety-np.pi,color=[0.1,1,0.1])
            #offset = offset+1+np.max(t_t)
            offsety = offsety-2*np.pi-np.pi/4
            
            plt.show()
        plt.xlabel('Time (s)')
    def jump_vs_heading(self,tphase='fsb_upper',plotall=True,ax=False,cmaps='plasma',samplepoints=20):
        from utils_plotting import uplt
        
        jdx = self.get_jumps()
        heading = self.ft2['ft_heading'].to_numpy()*-self.side
        phase = self.pdat['offset_'+tphase+'_phase']*-self.side
        print(self.side)
        pi = np.pi
        
        in_plume = np.zeros((jdx.shape[0],samplepoints,2))
        out_plume  =np.zeros((jdx.shape[0],samplepoints,2))
        new_time = np.linspace(0,samplepoints-1,samplepoints)
        for i,j in enumerate(jdx):
            x = heading[j[1]:j[2]]
            y = phase[j[1]:j[2]]
            if plotall:
                fig1, ax = plt.subplots()
                #plt.plot(x,y,cmap='rainbow')
                c = np.linspace(0,1,len(y))
                uplt.coloured_line(x,y,c,ax,cmap='plasma')
                plt.xlim([-pi,pi])
                plt.ylim([-pi,pi])
                plt.plot([-pi,pi],[-pi,pi],color='k',linestyle='--')
                plt.plot([0,0],[-pi,pi],color='k')
                plt.plot([-pi,pi],[0,0],color='k')
            old_time = np.linspace(0,samplepoints-1,len(y))
            x_int = np.interp(new_time,old_time,x)
            y_int = np.interp(new_time,old_time,y)
            out_plume[i,:,0] = x_int
            out_plume[i,:,1] = y_int
        
        omn = circmean(out_plume,axis=0,high=np.pi,low=-np.pi)
        if ax==False:
            fig1, ax = plt.subplots()
        c = np.linspace(0,1,len(omn))
        #uplt.coloured_line(omn[:,0],omn[:,1],c,ax,cmap=cmaps,linewidth=2)
        plt.plot(omn[:,0],omn[:,1],color='k',alpha=0.5)
        plt.xlim([-pi,pi])
        plt.ylim([-pi,pi])
        plt.plot([-pi,pi],[-pi,pi],color='k',linestyle='--')
        plt.plot([0,0],[-pi,pi],color='k')
        plt.plot([-pi,pi],[0,0],color='k')
        return omn
    def mean_jump_wedges(self,fsb_names=['fsb_upper','fsb_lower']):
        jumps = self.get_jumps()
        output_array = np.zeros((100,16,jumps.shape[0],len(fsb_names)))
        for inm,n in enumerate(fsb_names):
            twedge = self.pdat['wedges_offset_'+n]
            for ij,j in enumerate(jumps):
                tbef = np.arange(j[0],j[1],dtype='int')
                taf = np.arange(j[1],j[2],dtype='int')
                wbef = twedge[tbef,:]
                waf = twedge[taf,:]
                
                newtime = np.linspace(tbef[0],tbef[-1],50)
                for iw in range(16):
                    output_array[:50,iw,ij,inm] = np.interp(newtime,tbef,wbef[:,iw])
                
                newtime = np.linspace(taf[0],taf[-1],50)
                for iw in range(16):
                    output_array[50:,iw,ij,inm] = np.interp(newtime,taf,waf[:,iw])
        return output_array
    def phase_nulled_jump(self,bins=10,fsb_names=['fsb_upper']):
        jumps = self.get_jumps()
        output = np.zeros((16,bins*2,len(fsb_names)))
        
        for iw,w in enumerate(fsb_names):
            twedge = self.pdat['wedges_'+w]
            tphase = self.pdat['phase_'+w]
            tphase = tphase*-self.side
            if self.side==1:
                twedge = np.fliplr(twedge)
            wedgenull = ug.phase_nulling(twedge,tphase)
            befdata = np.zeros((16,bins,len(jumps)))
            afdata = np.zeros((16,bins,len(jumps)))
            for ij,j in enumerate(jumps):
                dxb = np.linspace(j[0],j[1],bins+1,dtype='int')
                
                
                dxa = np.linspace(j[1],j[2],bins+1,dtype='int')
                
                for b in range(bins):
                    bef_dx = np.arange(dxb[b],dxb[b+1])
                    if len(bef_dx)==0:
                        bef_dx = dxb[b]
                    af_dx = np.arange(dxa[b],dxa[b+1])
                    
                    befdata[:,b,ij] = np.nanmean(wedgenull[bef_dx,:],axis=0)
                    afdata[:,b,ij] = np.nanmean(wedgenull[af_dx,:],axis=0)
                
            output[:,:bins,iw] = np.mean(befdata,axis=2)
            output[:,bins:,iw] = np.mean(afdata,axis=2)
        return output
    
    def random_jump_wedge(self,fsb_names=['fsb_upper','fsb_lower']):
        jumps = self.get_jumps()
        jl = jumps.shape[0]
        rj = np.random.randint(0,jl)
        for inm,n in enumerate(fsb_names):
            twedge = self.pdat['wedges_offset_'+n]
            tphase = self.pdat['offset_'+n+'_phase'].to_numpy()
            j = jumps[rj,:]
            rdx = np.arange(j[0],j[2])
            tw = twedge[rdx,:]
            tp = tphase[rdx]
            if inm==0:
                output_array = np.zeros((tw.shape[0],tw.shape[1],len(fsb_names)))
                out_phase = np.zeros((len(tp),len(fsb_names)))
            out_phase[:,inm] = tp
            output_array[:,:,inm] = tw
            plumeoff = j[1]-j[0]
        return output_array,plumeoff,out_phase
        
    def specific_jump_wedge(self,jump,fsb_names=['fsb_upper','fsb_lower']):
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        x,y = self.fictrac_repair(x,y)
        for inm,n in enumerate(fsb_names):
            twedge = self.pdat['wedges_offset_'+n]
            tphase = self.pdat['offset_'+n+'_phase'].to_numpy()
            jump
            rdx = np.arange(jump[0],jump[2])
            tw = twedge[rdx,:]
            tp = tphase[rdx]
            if inm==0:
                output_array = np.zeros((tw.shape[0],tw.shape[1],len(fsb_names)))
                out_phase = np.zeros((len(tp),len(fsb_names)))
                traj = np.zeros((len(tp),2))
                traj[:,0] = x[rdx]
                traj[:,0] = traj[:,0]-traj[0,0]
                traj[:,1] = y[rdx]-y[jump[1]]
            out_phase[:,inm] = tp
            output_array[:,:,inm] = tw
            plumeoff = jump[1]-jump[0]
        return output_array,plumeoff,out_phase,traj
    def mean_jump_arrows(self,x_offset=0,fsb_names=['fsb_upper','fsb_lower'],ascale=50,jsize=3):
        
        ft2 = self.ft2
        pv2 = self.pv2
        jumps = ft2['jump']
        ins = ft2['instrip']
        x = ft2['ft_posx'].to_numpy()
        y = ft2['ft_posy'].to_numpy()
        times = pv2['relative_time']
        x,y = self.fictrac_repair(x,y)
        x = x+x_offset
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

        # Initialise arrays
        inplume_traj = np.zeros((50,len(this_j),2))
        outplume_traj = np.zeros((50,len(this_j),2))

        outplume_phase = np.zeros((50,len(this_j),3))
        inplume_phase = np.zeros((50,len(this_j),3))
        outplume_amp = np.zeros((50,len(this_j),3))
        inplume_amp = np.zeros((50,len(this_j),3))
        side_mult = side*-1
        
        plt.fill([-10+x_offset,x_offset,x_offset,-10+x_offset],[-50,-50,0,0],color=[0.8,0.8,0.8])
        plt.plot([x_offset,x_offset],[-50,0],linestyle='--',color='k',zorder=1)
        plt.fill([-13+x_offset,-jsize+x_offset,-jsize+x_offset,-13+x_offset],[50,50,0,0],color=[0.8,0.8,0.8])
        plt.plot([-jsize+x_offset,-jsize+x_offset],[50,0],linestyle='--',color='k',zorder=2)
        x = x*side_mult
        for i,j in enumerate(this_j):
            ex = exts-j
            ie = np.argmin(np.abs(ex))
            t_ent = ie+1
            sub_dx = exts[ie]
            phase = self.pdat['offset_eb_phase'].to_numpy()
            phase = phase.reshape(-1,1)
            for f in fsb_names:
                phase = np.append(phase,self.pdat['offset_' +f+'_phase'].to_numpy().reshape(-1,1),axis=1)
            phase = phase*side_mult
            amp = self.pdat['amp_eb']
            amp = amp.reshape(-1,1)
            for f in fsb_names:
                amp = np.append(amp,self.pdat['amp_'+f].reshape(-1,1),axis=1)
            # in plume
            ipdx = np.arange(ents[ie],sub_dx,step=1,dtype=int)
            old_time = ipdx-ipdx[0]
            ip_x = x[ipdx]
            ip_y = y[ipdx]
            ip_x = ip_x-ip_x[-1]+x_offset
            ip_y = ip_y-ip_y[-1]
            new_time = np.linspace(0,max(old_time),50)
            x_int = np.interp(new_time,old_time,ip_x)
            y_int = np.interp(new_time,old_time,ip_y)
            inplume_traj[:,i,0] = x_int
            inplume_traj[:,i,1] = y_int
            for p in range(len(fsb_names)+1):
                
                t_p = phase[ipdx,p]
                
                
                p_int = np.interp(new_time,old_time,t_p)
                inplume_phase[:,i,p] = p_int
                
                t_a = amp[ipdx,p]
                a_int = np.interp(new_time,old_time,t_a)
                inplume_amp[:,i,p] = a_int
            
            
            plt.plot(x_int,y_int,color='r',alpha=0.1,zorder=3)
            
            # out plume
            ipdx = np.arange(sub_dx,ents[t_ent],step=1,dtype=int)
            old_time = ipdx-ipdx[0]
            ip_x = x[ipdx]
            ip_y = y[ipdx]
            ip_x = ip_x-ip_x[0]+x_offset
            ip_y = ip_y-ip_y[0]
            new_time = np.linspace(0,max(old_time),50)
            x_int = np.interp(new_time,old_time,ip_x)
            y_int = np.interp(new_time,old_time,ip_y)
            outplume_traj[:,i,0] = x_int
            outplume_traj[:,i,1] = y_int
            for p in range(len(fsb_names)+1):
                t_p = phase[ipdx,p]
                p_int = np.interp(new_time,old_time,t_p)
                outplume_phase[:,i,p] = p_int
                
                t_a = amp[ipdx,p]
                a_int = np.interp(new_time,old_time,t_a)
                outplume_amp[:,i,p] = a_int
            
            plt.plot(x_int,y_int,color='k',alpha=0.1)
            
            tdx = np.arange(ents[ie],ents[t_ent],step=1,dtype='int')
            
        inmean_traj = np.mean(inplume_traj,axis=1)
        outmean_traj = np.mean(outplume_traj,axis=1)
        inmean_phase = circmean(inplume_phase,high=np.pi,low=-np.pi,axis=1)
        outmean_phase = circmean(outplume_phase,high=np.pi,low=-np.pi,axis=1)
        inmean_amp = np.mean(inplume_amp,axis=1)
        outmean_amp = np.mean(outplume_amp,axis=1)

        plt.plot(inmean_traj[:,0],inmean_traj[:,1],color='r',zorder=4)
        plt.plot(outmean_traj[:,0],outmean_traj[:,1],color='k',zorder=4)
        colours = np.array([[0.3,0.3,0.3],[0.3,0.3,1],[0.8,0.3,1]])

        tdx2 = np.arange(0,50,step=10,dtype=int)
        for p in range(3):
            for t in tdx2:
                xa = ascale*inmean_amp[t,p]*np.sin(inmean_phase[t,p])
                ya = ascale*inmean_amp[t,p]*np.cos(inmean_phase[t,p])
                plt.arrow(inmean_traj[t,0],inmean_traj[t,1],xa,ya,length_includes_head=True,head_width=1,color=colours[p,:],zorder=5)
                
                xa = ascale*outmean_amp[t,p]*np.sin(outmean_phase[t,p])
                ya = ascale*outmean_amp[t,p]*np.cos(outmean_phase[t,p])
                plt.arrow(outmean_traj[t,0],outmean_traj[t,1],xa,ya,length_includes_head=True,head_width=1,color=colours[p,:],zorder=5)
            xa = ascale*outmean_amp[-1,p]*np.sin(outmean_phase[-1,p])
            ya = ascale*outmean_amp[-1,p]*np.cos(outmean_phase[-1,p])
            plt.arrow(outmean_traj[-1,0],outmean_traj[-1,1],xa,ya,length_includes_head=True,head_width=1,color=colours[p,:],zorder=5)
        yl = [np.min(inmean_traj[:,1])-10,np.max(outmean_traj[:,1])+10]
        #plt.ylim(yl)
        #plt.xlim([-10,10])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        
    def mean_jump_heat(self,x_offset=0,regions=['eb','fsb_upper','fsb_lower']):
        # Mean heatmap phase aligned to plume factors
        from scipy.stats import circmean, circstd
        ft2 = self.ft2
        pv2 = self.pv2
        jumps = ft2['jump']
        ins = ft2['instrip']
        x = ft2['ft_posx'].to_numpy()
        y = ft2['ft_posy'].to_numpy()
        times = pv2['relative_time']
        x,y = self.fictrac_repair(x,y)
        x = x+x_offset
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

        # Initialise arrays
        inplume_traj = np.zeros((100,len(this_j),2))
        outplume_traj = np.zeros((100,len(this_j),2))
        
        outplume_phase = np.zeros((100,len(this_j),3))
        inplume_phase = np.zeros((100,len(this_j),3))
        
        outplume_amp = np.zeros((100,len(this_j),3))
        inplume_amp = np.zeros((100,len(this_j),3))
        
        outplume_heat = np.zeros((100,16,len(this_j),3)) 
        inplume_heat = np.zeros((100,16,len(this_j),3)) 
        side_mult = side*-1

        x = x*side_mult
        # Collate data for processing
        for ri,r in enumerate(regions):
            if ri==0:
                phase = self.pdat['offset_' + r + '_phase'].to_numpy()
                phase = phase.reshape(-1,1)
                
                heat = self.pdat['fit_wedges_' +r]
                heat = heat.reshape(-1,16,1)
                
                amp = self.pdat['amp_' +r]
                amp = amp.reshape(-1,1)
            else:
                phase = np.append(phase,self.pdat['offset_' + r + '_phase'].to_numpy().reshape(-1,1),axis=1)
                heat = np.append(heat,self.pdat['fit_wedges_'+ r ].reshape(-1,16,1),axis=2)
                amp = np.append(amp,self.pdat['amp_' +r ].reshape(-1,1),axis=1)
                
        # Flip data all to one side        
        phase = phase*side_mult
        if side_mult<0:
            heat = np.fliplr(heat)
            
        # Run through entry/exits    
        
        for i,j in enumerate(this_j):
            ex = exts-j
            ie = np.argmin(np.abs(ex))
            t_ent = ie+1
            sub_dx = exts[ie]
            # in plume
            ipdx = np.arange(ents[ie],sub_dx,step=1,dtype=int)
            old_time = ipdx-ipdx[0]
            ip_x = x[ipdx]
            ip_y = y[ipdx]
            ip_x = ip_x-ip_x[-1]+x_offset
            ip_y = ip_y-ip_y[-1]
            new_time = np.linspace(0,max(old_time),100)
            x_int = np.interp(new_time,old_time,ip_x)
            y_int = np.interp(new_time,old_time,ip_y)
            inplume_traj[:,i,0] = x_int
            inplume_traj[:,i,1] = y_int
            for p in range(len(regions)):
                t_p = phase[ipdx,p]
                p_int = np.interp(new_time,old_time,t_p)
                inplume_phase[:,i,p] = p_int
                
                t_a = amp[ipdx,p]
                a_int = np.interp(new_time,old_time,t_a)
                inplume_amp[:,i,p] = a_int
                
                t_h = heat[ipdx,:,p]
                for it in range(16):
                    h_int = np.interp(new_time,old_time,t_h[:,it])
                    inplume_heat[:,it,i,p] = h_int
            
            
            
            # out plume
            print(sub_dx)
            print(ents[t_ent])
            ipdx = np.arange(sub_dx,ents[t_ent],step=1,dtype=int)
            old_time = ipdx-ipdx[0]
            ip_x = x[ipdx]
            ip_y = y[ipdx]
            ip_x = ip_x-ip_x[0]+x_offset
            ip_y = ip_y-ip_y[0]
            new_time = np.linspace(0,max(old_time),100)
            x_int = np.interp(new_time,old_time,ip_x)
            y_int = np.interp(new_time,old_time,ip_y)
            outplume_traj[:,i,0] = x_int
            outplume_traj[:,i,1] = y_int
            for p in range(len(regions)):
                t_p = phase[ipdx,p]
                p_int = np.interp(new_time,old_time,t_p)
                outplume_phase[:,i,p] = p_int
                
                t_a = amp[ipdx,p]
                a_int = np.interp(new_time,old_time,t_a)
                outplume_amp[:,i,p] = a_int
                
                t_h = heat[ipdx,:,p]
                for it in range(16):
                    h_int = np.interp(new_time,old_time,t_h[:,it])
                    outplume_heat[:,it,i,p] = h_int
                    
        inmean_traj = np.mean(inplume_traj,axis=1)
        outmean_traj = np.mean(outplume_traj,axis=1)
        inmean_phase = circmean(inplume_phase,high=np.pi,low=-np.pi,axis=1)
        outmean_phase = circmean(outplume_phase,high=np.pi,low=-np.pi,axis=1)
        inmean_amp = np.mean(inplume_amp,axis=1)
        outmean_amp = np.mean(outplume_amp,axis=1)
        inmean_heat = np.mean(inplume_heat,axis=2)
        outmean_heat = np.mean(outplume_heat,axis=2)
        # Interploate data onto y axis
        
        ptraj = np.append(inmean_traj,outmean_traj,axis=0)
        
        y = ptraj[:,1]
        f,a = plt.subplots(1,4)
        a[0].fill([-10+x_offset,x_offset,x_offset,-10+x_offset],[-50,-50,0,0],color=[0.8,0.8,0.8])
        a[0].plot([x_offset,x_offset],[-50,0],linestyle='--',color='k',zorder=1)
        a[0].fill([-13+x_offset,-3+x_offset,-3+x_offset,-13+x_offset],[50,50,0,0],color=[0.8,0.8,0.8])
        a[0].plot([-3+x_offset,-3+x_offset],[50,0],linestyle='--',color='k',zorder=2)
        a[0].plot(inmean_traj[:,0],inmean_traj[:,1],zorder=3,color='r')
        a[0].plot(outmean_traj[:,0],outmean_traj[:,1],zorder=3,color='k')
        a[0].set_aspect('equal', adjustable='box')
        a[0].set_ylim(min(y),max(y))
        a[0].set_title('Mean trajectory')  
        yplt = np.arange(min(y),max(y),step=1)
        yp = np.linspace(0,len(yplt)-1,len(yplt))
        for ir,r in enumerate(regions): 
            pltmat = np.append(np.flipud(outmean_heat[:,:,ir]),np.flipud(inmean_heat[:,:,ir]),axis=0)
            pltmat = np.flipud(pltmat)
            pltmat_int = np.zeros((len(yplt),16))
            for ip in range(16):   
                pltmat_int[:,ip] = np.interp(yplt,y,pltmat[:,ip])
            pltmat_int = np.flipud(pltmat_int)
            a[ir+1].imshow(pltmat_int, interpolation='None',aspect='auto',cmap='Blues',vmax=np.nanpercentile(pltmat[:],99),vmin=np.nanpercentile(pltmat[:],5))
            a[ir+1].set_aspect('equal', adjustable='box')
            a[ir+1].set_yticks([])
            a[ir+1].set_xticks([0,3.5,7,10.5,15])
            a[ir+1].set_xticklabels([-180,-90,0,90,180])
            a[ir+1].set_title(r)
            pltphase = np.append(np.flipud(outmean_phase[:,ir]),np.flipud(inmean_phase[:,ir]),axis=0)
            pltphase = np.flipud(pltphase)
            pltphase_int = np.interp(yplt,y,pltphase)
            pltphase_int = np.flipud(pltphase_int)
            pltphase = 15.5*(pltphase_int+np.pi)/(2*np.pi)
            a[ir+1].plot([7, 7],[yp[0], yp[-1]],color='w',linestyle='--')
            a[ir+1].plot(pltphase,yp,color='k')
       
        # for i,j in enumerate(this_j):
        #     y = np.append(inplume_traj[:,i,1],outplume_traj[:,i,1],axis=0)
            
        #     f,a = plt.subplots(1,4)
        #     a[0].fill([-10+x_offset,x_offset,x_offset,-10+x_offset],[-50,-50,0,0],color=[0.8,0.8,0.8])
        #     a[0].plot([x_offset,x_offset],[-50,0],linestyle='--',color='k',zorder=1)
        #     a[0].fill([-13+x_offset,-3+x_offset,-3+x_offset,-13+x_offset],[50,50,0,0],color=[0.8,0.8,0.8])
        #     a[0].plot([-3+x_offset,-3+x_offset],[50,0],linestyle='--',color='k',zorder=2)
        #     a[0].plot(inplume_traj[:,i,0],inplume_traj[:,i,1],zorder=3,color='r')
        #     a[0].plot(outplume_traj[:,i,0],outplume_traj[:,i,1],zorder=3,color='k')
        #     a[0].set_aspect('equal', adjustable='box')
        #     print(min(y))
        #     a[0].set_ylim(min(y),max(y))
        #     a[0].set_title('Mean trajectory')  
        #     yplt = np.linspace(min(y),max(y),100)
        #     yp = np.linspace(0,len(yplt)-1,len(yplt))
        #     for ir,r in enumerate(regions): 
        #         pltmat = np.append(np.flipud(outplume_heat[:,:,i,ir]),np.flipud(inplume_heat[:,:,i,ir]),axis=0)
        #         pltmat = np.flipud(pltmat)
        #         pltmat_int = np.zeros((len(yplt),16))
        #         for ip in range(16):   
        #             pltmat_int[:,ip] = np.interp(yplt,y,pltmat[:,ip])
        #         pltmat_int = np.flipud(pltmat_int)
        #         a[ir+1].imshow(np.flipud(pltmat), interpolation='None',aspect='auto',cmap='Blues',vmax=np.nanpercentile(pltmat[:],99),vmin=np.nanpercentile(pltmat[:],5))
        #         a[ir+1].set_aspect('equal', adjustable='box')
        #         a[ir+1].set_yticks([])
        #         a[ir+1].set_xticks([0,3.5,7,10.5,15])
        #         a[ir+1].set_xticklabels([-180,-90,0,90,180])
        #         a[ir+1].set_title(r)
        #         pltphase = np.append(np.flipud(outplume_phase[:,i,ir]),np.flipud(inplume_phase[:,i,ir]),axis=0)
        #         pltphase = np.flipud(pltphase)
        #         pltphase_int = np.interp(yplt,y,pltphase)
        #         pltphase_int = np.flipud(pltphase_int)
        #         pltphase_int = 15.5*(pltphase_int+np.pi)/(2*np.pi)
        #         a[ir+1].plot([7, 7],[yp[0], yp[-1]],color='w',linestyle='--')
        #        # a[ir+1].plot(pltphase,yp,color='k')
        #         a[ir+1].set_aspect('equal', adjustable='box')
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
        
    def point2point_heat(self,start,stop,regions=['eb','fsb_upper','fsb_lower'],arrowpoint='entry',toffset=-1):
        # Function will plot trajectory with arrows plus heatmaps of regions
        # The timepoints of arrows will be highlighted
        
        from matplotlib.colors import LinearSegmentedColormap


        colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
        cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)
        ft2 = self.ft2
        pv2 = self.pv2
        jumps = ft2['jump']
        bumps = ft2['bump']
        jumps = jumps-np.mod(jumps,3)#gets rid of half jumps
        
        
        
        # Work out jump side
        jd = np.diff(jumps)
        jn = np.where(np.abs(jd)>0)[0]
        jkeep = np.where(np.diff(jn)>1)[0]
        jn = jn[jkeep]
        jns = np.sign(jd[jn])

        time_threshold = 60
        # Pick the most common side
        v,c = np.unique(jns,return_counts=True)
        side = v[np.argmax(c)]
        side_mult = side*-1
        jumps = jumps*side_mult
        
        
        ins = ft2['instrip']
        x = ft2['ft_posx'].to_numpy()*side_mult
        y = ft2['ft_posy'].to_numpy()
        
        pst = np.where(ins==1)[0][0]
        heading = ft2['ft_heading'].to_numpy()*side_mult
        times = pv2['relative_time']
        tres = np.mean(np.diff(times))
        tint = int(np.round(toffset/tres))
        x,y = self.fictrac_repair(x,y)
        #x,y = self.bumpstraighten(x,y,ft2['ft_heading'])
        
        f,a = plt.subplots(1,1+len(regions))
        t_x = x[start:stop]
        t_y = y[start:stop]
        
        x_off= t_x[0]
        t_x = t_x-x[pst]
        t_y = t_y-x[pst]
        
        
        
        
        t_j = jumps[start:stop]
        
        t_ins = ins[start:stop]
        t_id = np.diff(t_ins)
        pst = np.where(t_id>0)[0]+1 
        ped = np.where(t_id<0)[0]+1
        
        
        
        
        
        
        if len(pst)<1:
            print('nooot')
            a[0].plot(t_x,t_y,color='k')

        else:
            if pst[0]>ped[0]:
                print('True')
                pst = np.append(0,pst)
                
            if len(pst)>len(ped):
                ped = np.append(ped,len(t_j))
            print(pst)
            print(ped)
            t_j = np.round(t_j)
            uj = np.unique(t_j)
            uj = uj[~np.isnan(uj)]
            iu = np.argsort(-uj)
            uj = uj[iu]
            for ij, j in enumerate(uj):
                fx = [j-5,j+5]
                print(j)
                if ij==0:
                    fy1 = min(t_y[t_j==j])
                else:
                    fy1 = fy2.copy()
                print('j', j)
                fy2 = max(t_y[t_j==j])
                a[0].fill_between(fx,[fy1,fy1],[fy2,fy2],color=[0.7,0.7,0.7])
                a[0].plot([fx[1],fx[1]],[fy1,fy2],color='k',linestyle='--')
               
            
            for i,p in enumerate(pst):
                a[0].plot(t_x[p:ped[i]],t_y[p:ped[i]],color=[1,0.3,0.3])
            for i in range(len(ped)-1):
                a[0].plot(t_x[ped[i]-1:pst[i+1]+1],t_y[ped[i]-1:pst[i+1]+1],color='k')
                
                
            a[0].plot(t_x[:pst[0]],t_y[:pst[0]],color='k')
            
        a[0].set_aspect('equal', adjustable='box')
        a[0].set_ylim(min(t_y),max(t_y))
        a[0].set_yticks([])
        colours = np.array([[0,0,0],[0.3,0.3,1],[0.8,0.3,1]])
        heat_cols = ['Greys','Blues','Purples']
        if isinstance(arrowpoint,np.ndarray):
            t_arrows = arrowpoint
        elif arrowpoint=='entry':
            t_arrows = pst[pst>0]
        for it,t in enumerate(t_arrows):
            print(t)
            a[0].text(t_x[t+tint]+1,t_y[t+tint],str(it+1))
            for i,r in enumerate(regions):
                t_phase = self.pdat['offset_'+r+'_phase'][start:stop].to_numpy()*side_mult
                t_phase = t_phase
                t_amp = self.pdat['amp_'+r ][start:stop]
                tp = t_phase[t+tint]
                ta = t_amp[t+tint]
                ax = np.sin(tp)*10
                ay = np.cos(tp)*10
                a[0].arrow(t_x[t+tint],t_y[t+tint],ax,ay,color=colours[i],length_includes_head=True,head_width=1,zorder=10)
            a[0].scatter(t_x[t+tint],t_y[t+tint],color=[0.3,1,0.3],marker='*',zorder=10)
                
        for i,r in enumerate(regions):
            #heat = self.pdat['fit_wedges_' +r][start:stop,:]
            
            heat = self.pdat['wedges_offset_' +r][start:stop,:]
            if side_mult<0:
                heat = np.fliplr(heat)
            t_phase = self.pdat['offset_'+r+'_phase'][start:stop].to_numpy()*side_mult
            
            t_plot = 16*(t_phase+np.pi)/(2*np.pi)-0.5
            #a[i+1].plot(t_plot,np.arange(len(heat)))
            if i==0:
                a[i+1].imshow(heat, interpolation='None',aspect='auto',cmap=heat_cols[i],vmax=np.nanpercentile(heat[:],95),vmin=np.nanpercentile(heat[:],25))
            else:
                a[i+1].imshow(heat, interpolation='None',aspect='auto',cmap=heat_cols[i],vmax=np.nanpercentile(heat[:],95),vmin=np.nanpercentile(heat[:],25))
            a[i+1].set_xlim([-1,16])
            a[i+1].set_title(r)
            
            for ie,e in enumerate(pst):
                
                ent = e
                ext = ped[ie]
                a[i+1].plot([-0.5,15.5,15.5,-0.5,-0.5],[ent,ent,ext,ext,ent],color='r')
            a[i+1].plot([7.5,7.5],[0,len(heat)],color='w',linestyle='--')
            a[i+1].set_yticks([])
            a[i+1].set_ylim([0,len(heat)])
            a[i+1].set_xticks([-0.5,3.5,7.5,11.5,15.5],labels=[-180,-90,0,90,180])
            for t in t_arrows:
                it = t_phase[t+tint]
                it = 16*(it+np.pi)/(2*np.pi)-0.5
                a[i+1].scatter([it-0.5],[t+tint],color=[0.3,1,0.3],marker='*')
        a[1].set_yticks(np.transpose(t_arrows+tint),labels=np.arange(1,len(t_arrows)+1))
        
    def all_jump_heat(self,regions=['eb','fsb_upper','fsb_lower'],x_offset=0):
        plt.close('all')
        from scipy.stats import circmean, circstd
        ft2 = self.ft2
        pv2 = self.pv2
        jumps = ft2['jump']
        ins = ft2['instrip']
        x = ft2['ft_posx'].to_numpy()
        y = ft2['ft_posy'].to_numpy()
        heading = ft2['ft_heading'].to_numpy()
        times = pv2['relative_time']
        x,y = self.fictrac_repair(x,y)
       
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

        side_mult = side*-1

        x = x*side_mult
        # Collate data for processing
        for ri,r in enumerate(regions):
            if ri==0:
                phase = self.pdat['offset_' + r + '_phase'].to_numpy()
                phase = phase.reshape(-1,1)
                
                heat = self.pdat['fit_wedges_' +r]
                heat = heat.reshape(-1,16,1)
                
                amp = self.pdat['amp_' +r]
                amp = amp.reshape(-1,1)
            else:
                phase = np.append(phase,self.pdat['offset_' + r + '_phase'].to_numpy().reshape(-1,1),axis=1)
                heat = np.append(heat,self.pdat['fit_wedges_'+ r ].reshape(-1,16,1),axis=2)
                amp = np.append(amp,self.pdat['amp_' +r ].reshape(-1,1),axis=1)
                
        # Flip data all to one side        
        phase = phase*side_mult
        if side_mult<0:
            heat = np.fliplr(heat)
            
        # Run through entry/exits    
        for i,j in enumerate(this_j):
            ex = exts-j
            ie = np.argmin(np.abs(ex))
            t_ent = ie+1
            sub_dx = exts[ie]
            # in and out plume
            ipdx = np.arange(ents[ie],ents[t_ent],step=1,dtype=int)
            pon = np.where(ipdx==sub_dx)[0]
            old_time = ipdx-ipdx[0]
            ip_x = x[ipdx]
            ip_y = y[ipdx]
            ip_x = ip_x-ip_x[ipdx==sub_dx]
            ip_y = ip_y-ip_y[ipdx==sub_dx]
            f,a = plt.subplots(1,4)
            a[0].fill([-10+x_offset,x_offset,x_offset,-10+x_offset],[-100,-100,0,0],color=[0.8,0.8,0.8])
            a[0].plot([x_offset,x_offset],[-50,0],linestyle='--',color='k',zorder=1)
            a[0].fill([-13+x_offset,-3+x_offset,-3+x_offset,-13+x_offset],[100,100,0,0],color=[0.8,0.8,0.8])
            a[0].plot([-3+x_offset,-3+x_offset],[50,0],linestyle='--',color='k',zorder=2)
            a[0].plot(ip_x,ip_y,zorder=3,color='k')
            a[0].set_aspect('equal', adjustable='box')
            a[0].set_ylim(min(ip_y),max(ip_y))
            a[0].set_title('Trajectory')
            for p in range(len(regions)):
                t_p = phase[ipdx,p]
                t_a = amp[ipdx,p]
                t_h = heat[ipdx,:,p]
                t_head = heading[ipdx]
                t_head = 15.5*(t_head+np.pi)/(2*np.pi)
                t_p = 15.5*(t_p+np.pi)/(2*np.pi)
                a[p+1].imshow(np.flipud(t_h), interpolation='None',aspect='auto',cmap='Blues',vmax=np.nanpercentile(t_h[:],99),vmin=np.nanpercentile(t_h[:],5))
                p_y = np.arange(0,len(t_p),step=1)
                a[p+1].plot(np.flipud(t_head),p_y,color='r')
                a[p+1].plot([0,16],[len(t_p)-pon,len(t_p)-pon],linestyle='--',color='k')
                a[p+1].plot(np.flipud(t_p),p_y,color='g')
                a[p+1].set_title(regions[p])
                
    def plume_meno_comp(self,regions=['eb','fsb_upper','fsb_lower'],diff_phase = True,diff_val='heading'):
        ft2 = self.ft2
        ft = self.ft
        pv2 = self.pv2
        colours = np.array([[0,0,0],[0.3,0.3,1],[1,0.3,1]])
        stimon = np.where(ft2['instrip']==1)[0][0]
        meno_period = np.arange(0,stimon,dtype=int)
        heading = ft2['ft_heading']
        radbins = np.linspace(-1,1,num=20)*np.pi
        pltbins = 180*radbins/np.pi
        pltbins = pltbins[1:]-(np.mean(np.diff(pltbins))/2)
        plt.xticks([-180,-90,0,90,180])
        plt.xlabel('Phase (deg)')
        plt.ylabel('Probability')
        plt.title('Amenotaxis')
        
        
        # Plume jump returns
        t_return = 3
        
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
        side_mult = side*-1
        heading= heading*side_mult
        
        for i, r in enumerate(regions):
            t_phase = self.pdat['offset_' +r+ '_phase']*side_mult
            if diff_phase:
                if diff_val=='heading':
                    p_diff = t_phase-heading
                else:
                    p_diff = t_phase-(self.pdat['offset_'+ diff_val+'_phase']*side_mult)
                # Brings stats back to -pi pi
                p_cos = np.cos(p_diff)
                p_sin = np.sin(p_diff)
                p_diffa = np.arctan2(p_sin,p_cos)
            else:
                p_diffa = t_phase
            
            h = np.histogram(p_diffa,bins = radbins,density=True)
            ph = h[0]
            ph = ph/np.sum(ph)
            plt.plot(pltbins,ph,color= colours[i,:])
            if i==0:
                meno_array = ph
                meno_array = np.reshape(meno_array,(-1,1))
            else:
                
                meno_array = np.append(meno_array,ph.reshape((-1,1)),axis=1)
            
        
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
        
        tres = np.mean(np.diff(times))
        tnums = np.round(t_return/tres)
        plt.figure()
        
        for i,r in enumerate(regions):
            t_phase = self.pdat['offset_' +r+ '_phase']*side_mult
            if diff_phase:
                if diff_val=='heading':
                    p_diff = t_phase-heading
                else:
                    p_diff = t_phase-(self.pdat['offset_'+ diff_val+'_phase']*side_mult)
            else:
                p_diff = t_phase
            h_pdiff = np.array([])
            for ij,j in enumerate(this_j):
                ex = exts-j
                ie = np.argmin(np.abs(ex))
                t_ent = ie+1
                ist = np.max([ents[t_ent]-tnums, exts[ie]])
                ipdx = np.arange(ist,ents[t_ent],step=1,dtype=int)
                
                h_pdiff = np.append(h_pdiff,p_diff[ipdx])
            p_cos = np.cos(h_pdiff)
            p_sin = np.sin(h_pdiff)
            p_diffa = np.arctan2(p_sin,p_cos)
            h = np.histogram(p_diffa,bins=radbins,density=True)
            ph = h[0]
            ph = ph/np.sum(ph)
            plt.plot(pltbins,ph,color= colours[i,:])
            if i==0:
                et_array = ph
                et_array = np.reshape(et_array,(-1,1))
            else:
                et_array = np.append(et_array,ph.reshape((-1,1)),axis=1)
        plt.title('Plume jump returns')
        plt.xlabel('Phase (deg)')
        plt.ylabel('Probability')
        plt.xticks([-180,-90,0,90,180])
        
        plt.figure()
        t_exit = 0.25
        tnums = np.round(t_exit/tres)
        lenmin = np.round(0/tres)
        for i,r in enumerate(regions):
            t_phase = self.pdat['offset_' +r+ '_phase']*side_mult
            if diff_phase:
                if diff_val=='heading':
                    p_diff = t_phase-heading
                else:
                    p_diff = t_phase-(self.pdat['offset_'+ diff_val+'_phase']*side_mult)
            else:
                p_diff = t_phase
            h_pdiff = np.array([])
            for ij,j in enumerate(this_j):
                ex = exts-j
                ie = np.argmin(np.abs(ex))
                print(exts[ie],ents[ie])
                if (exts[ie]-ents[ie])<lenmin:
                    print('Too short!!')
                    continue
                # Get part just before exit
                ist = np.max([exts[ie]-tnums,ents[ie]])
                ipdx = np.arange(ist,exts[ie],step=1,dtype=int)
                
                h_pdiff = np.append(h_pdiff,p_diff[ipdx])
            p_cos = np.cos(h_pdiff)
            p_sin = np.sin(h_pdiff)
            p_diffa = np.arctan2(p_sin,p_cos)
            h = np.histogram(p_diffa,bins=radbins,density=True)
            ph = h[0]
            ph = ph/np.sum(ph)
            plt.plot(pltbins,ph,color= colours[i,:])
            if i==0:
                et_array_ex = ph
                et_array_ex = np.reshape(et_array_ex,(-1,1))
            else:
                et_array_ex = np.append(et_array_ex,ph.reshape((-1,1)),axis=1)
        plt.title('Plume jump exits')
        plt.xlabel('Phase (deg)')
        plt.ylabel('Probability')
        plt.xticks([-180,-90,0,90,180])
        
        return meno_array,et_array,et_array_ex,pltbins
        # Aim is to make a comparison between EPG and 
    def mean_phase_arrow(self,tbef=5,taf=5,phase='offset_fsb_phase',stype='On'):
        eb_phase = self.pdat['offset_eb_phase']
        fsb_phase = self.pdat[phase]
        #eb_phase = self.phase_eb
        #fsb_phase = self.phase
        strip = self.ft2['instrip'].to_numpy()
        sdiff = np.diff(strip)
        ts = self.pv2['relative_time'].to_numpy()
        tinc = np.mean(np.diff(ts))
        if stype=='On':
            son = np.where(sdiff>0)[0]+1
        elif stype=='Off':
            son = np.where(sdiff<0)[0]+1
        idx_bef = int(np.round(float(tbef)/tinc))
        idx_af = int(np.round(float(taf)/tinc))
        son = son[1:]
        mn_mat = np.zeros((len(son),idx_bef+idx_af+1))
        
        for i,s in enumerate(son):
            idx_array = np.arange(s-idx_bef-1,s+idx_af,dtype= int)
            if idx_array[-1]> len(eb_phase):
                nsum = np.sum(idx_array>len(eb_phase))
                idx_array = idx_array[idx_array<len(fsb_phase)]
                mn_mat[i,:-nsum] = eb_phase[idx_array]
            else:
                mn_mat[i,:] = eb_phase[idx_array]
        plt_mn = circmean(mn_mat,axis=0,high=np.pi,low=-np.pi)
        t = np.linspace(-tbef,taf,idx_bef+idx_af+1)
        mult = 3;
        
        plt.figure()
        plt.plot([-3, 3],[0,0],linestyle='--',color='k')
        for i,m in enumerate(plt_mn):
            if np.mod(i,2)==0:
                x = mult*np.sin(m)
                y = mult*np.cos(m)
                plt.arrow(0,t[i],x,y)
            
            
        mn_mat2 = np.zeros((len(son),idx_bef+idx_af+1))
        for i,s in enumerate(son):
            idx_array = np.arange(s-idx_bef-1,s+idx_af,dtype= int)
            if idx_array[-1]> len(fsb_phase):
                nsum = np.sum(idx_array>len(fsb_phase))
                idx_array = idx_array[idx_array<len(fsb_phase)]
                mn_mat2[i,:-nsum] = fsb_phase[idx_array]
            else:
                mn_mat2[i,:] = fsb_phase[idx_array]
        plt.xlim([-3,3])  
        plt.ylim([min(t)-3,max(t)+3])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        #plt.figure()
        #plt.plot([-3, 3],[0,0],linestyle='--',color='k')        
        plt_mn = circmean(mn_mat2,axis=0,high=np.pi,low=-np.pi)
        t = np.linspace(-tbef,taf,idx_bef+idx_af+1)
        
        for i,m in enumerate(plt_mn):
            if np.mod(i,2)==0:
                x = mult*np.sin(m)
                y = mult*np.cos(m)
                plt.arrow(0,t[i],x,y,color=[0.6, 0.6, 1])
            
        plt.xlim([-3,3])
        plt.ylim([min(t)-3,max(t)+3])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.ylabel('Time (s)')
        plt.show()
    def plot_traj_arrow_peaks(self,region):
        from scipy import signal as sg
        from scipy import stats
        angles = np.linspace(-np.pi,np.pi,16)
        try:
            phase = self.pdat['offset_'+region+'_phase'].to_numpy()
        except:
            phase = self.pdat['offset_'+region+'_phase']
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        x,y = self.fictrac_repair(x,y)
        instrip = self.ft2['instrip'].to_numpy()
        is1 =np.where(instrip)[0][0]
        is2 = np.where(instrip)[0]
        wedges = self.pdat['wedges_'+region]
        weds = np.sum(wedges*np.sin(angles),axis=1)
        wedc = np.sum(wedges*np.cos(angles),axis=1)
        pva  = np.sqrt(weds**2+wedc**2)
        p0 = np.mean(pva[pva<np.percentile(pva,10)])
        pva = (pva-p0)/p0
        pva = pva/np.max(pva)
        
        pvsmooth = sg.savgol_filter(pva,40,3)
        pvstd = np.std(pvsmooth)
        peaks = sg.find_peaks(pvsmooth,prominence=pvstd) #height=pvstd,
        
        plt.figure()
        tt = self.pv2['relative_time'].to_numpy()
        heading = self.ft2['ft_heading'].to_numpy()
        plt.plot(tt,pva,color='k')
        plt.plot(tt,instrip-1,color='r')
        plt.plot(tt,pvsmooth)
        p_phases = np.zeros(len(peaks[0]))
        h_sum = np.zeros(len(peaks[0]))
        idiff = np.diff(instrip)
        iw = np.append(0,np.where(idiff<0)[0])
        
        for i, p in enumerate(peaks[0]):
           # print(p)
            # dx = np.cos(peakphase[p])
            # dy = np.sin(peakphase[p])
            # plt.arrow(p,pvsmooth[p],dx,dy)
            an = stats.circmean(phase[p-5:p+5],low=-np.pi,high=np.pi)
            p_phases[i] = an
            an = 180*an/np.pi
            an = np.round(an)
            
            try: 
                nodour = np.min(is2[is2>p])
                tx = x[p]
                nx = x[nodour]
                side = np.sign(tx-nx)
                if side<0:
                    plt.text(tt[p],pvsmooth[p]+0.2,str(an)+ ' ls',ha='center')
                    plt.scatter(tt[p],pvsmooth[p],marker='>',zorder=10,color='r')
                else:
                    plt.text(tt[p],pvsmooth[p]+0.2,str(an) +' rs',ha='center')
                    plt.scatter(tt[p],pvsmooth[p],marker='<',zorder=10,color='r')
            except:
                print('end of odour')
                
            
            nodour2 = np.max(iw[iw<p])
            th = heading[nodour2:p]
            thu = fn.unwrap(th)
            #h_sum[i] = fn.wrap(thu[-1])
            
                
        plt.figure()
        
        plt.plot(x,y,color='k')
        plt.scatter(x[is2],y[is2],color=[0.7,0.7,0.7])
        for i,p in enumerate(peaks[0]):
            parray = np.arange(p-50,p,5)
            ip = p
            alpha = np.linspace(0.3,1,len(parray))
            # for i1,ip in enumerate(parray):
           
            #     xa = 50*pvsmooth[ip]*np.sin(phase[ip])
            #     ya = 50*pvsmooth[ip]*np.cos(phase[ip])
            #     plt.arrow(x[ip],y[ip],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1],alpha=alpha[i1])
            
            xa = 50*pvsmooth[ip]*np.sin(phase[ip])
            ya = 50*pvsmooth[ip]*np.cos(phase[ip])
            plt.arrow(x[ip],y[ip],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1],alpha=1)
            
            # ta = -h_sum[i]
            # xa = 50*pvsmooth[p]*np.sin(ta)
            # ya = 50*pvsmooth[p]*np.cos(ta)
            # plt.arrow(x[p],y[p],xa,ya,length_includes_head=True,head_width=1,color='k')
        
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        
        
        
        
        
        
    def plot_traj_arrow(self,phase,amp,a_sep= 20,traindat=False):
        try:
            phase_eb = self.pdat['offset_eb_phase'].to_numpy()
        except:
            phase_eb = self.pdat['offset_pb_phase'].to_numpy()
        #phase_eb = self.phase_eb
        amp_eb = self.amp_eb.copy()
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        
        
        x,y = self.fictrac_repair(x,y)
        instrip = self.ft2['instrip'].to_numpy()
        if traindat:
            mfc = self.ft2['mfc2_stpt'].to_numpy()>0
            it = self.ft2['intrain'].to_numpy()>0
        try:    
            is1 =np.where(instrip)[0][0]
        except:
            instrip = self.ft2['mfc3_stpt'].to_numpy()>0
            is1 = np.where(instrip)[0][0]
        dist = np.sqrt(x**2+y**2)
        dist = dist-dist[0]
        plt.figure()
        
        x = x[is1:]
        y = y[is1:]
        dist = dist[is1:]
        
        instrip = instrip[is1:]
        if traindat:
            it = it[is1:]
            mfc = mfc[is1:]
        phase = phase[is1:]
        phase_eb = phase_eb[is1:]
        amp = amp[is1:]
        amp_eb = amp_eb[is1:]
        
        
        if traindat:
            plt.scatter(x[it],y[it],color=[0.8,0.2,0.2])
            ito = np.logical_and(it,mfc)
            
        plt.scatter(x[instrip>0],y[instrip>0],color=[0.6,0.6,0.6])
        if traindat:
            plt.scatter(x[ito],y[ito],color=[0.6,0.6,0.6])
        
        plt.plot(x,y,color='k')
        t_sep = a_sep
        for i,d in enumerate(dist):
            if np.abs(d-t_sep)>a_sep:
                t_sep = d
                
                xa = 50*amp_eb[i]*np.sin(phase_eb[i])
                ya = 50*amp_eb[i]*np.cos(phase_eb[i])
                plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color=[0.1,0.1,0.1])
                
                xa = 50*amp[i]*np.sin(phase[i])
                ya = 50*amp[i]*np.cos(phase[i])
                plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    def plot_traj_arrow_jump_amp(self,phase,amp,a_sep= 20,traindat=False):
        from analysis_funs.regression import fci_regmodel
        fci = fci_regmodel(amp,self.ft2,self.pv2)
        cmin = np.percentile(amp,10)
        cmax = np.percentile(amp,90)
        fci.example_trajectory_jump(amp,self.ft,cmin=cmin,cmax=cmax)
        
        try:
            phase_eb = self.pdat['offset_eb_phase'].to_numpy()
        except:
            phase_eb = self.pdat['offset_pb_phase'].to_numpy()
        #phase_eb = self.phase_eb
        amp_eb = self.amp_eb.copy()
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        
        
        x,y = self.fictrac_repair(x,y)
        instrip = self.ft2['instrip'].to_numpy()
        
            
        is1 =np.where(instrip)[0][0]
        dist = np.sqrt(np.diff(x)**2+np.diff(y)**2)
        #dist = dist-dist[0]
        dist = np.append(0,dist)
        
        x = x[is1:]
        y = y[is1:]
        x=x-x[0]
        y = y-y[0]
        dist = dist[is1:]
        
        instrip = instrip[is1:]
        
        phase = phase[is1:]
        phase_eb = phase_eb[is1:]
        amp = amp[is1:]
        amp_eb = amp_eb[is1:]
        
        
        
            
        #plt.scatter(x[instrip>0],y[instrip>0],color=[0.6,0.6,0.6])
        
        t_sep = a_sep
        dsum = 0
        for i,d in enumerate(dist):
            dsum += d
            if np.abs(dsum)>a_sep:
                dsum = 0
                
                xa = 50*amp_eb[i]*np.sin(phase_eb[i])
                ya = 50*amp_eb[i]*np.cos(phase_eb[i])
                plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color=[0.1,0.1,0.1])
               
                xa = 10*amp[i]*np.sin(phase[i])
                ya = 10*amp[i]*np.cos(phase[i])
                plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
            
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        
    def plot_traj_cond(self,phase,amp,cond):
        phase_eb = self.pdat['offset_eb_phase']
        
        amp_eb = self.amp_eb
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        instrip = self.ft2['instrip'].to_numpy()
        
        dist = np.sqrt(x**2+y**2)
        dist = dist-dist[0]
        plt.figure()
        plt.scatter(x[instrip>0],y[instrip>0],color=[0.6,0.6,0.6])
        plt.plot(x,y,color='k')
        if cond=='entry_exit':
            dstrip = np.diff(instrip)
            dx = np.where(np.abs(dstrip)>0)[0]+1
        
        for i in dx:
            
                
            xa = 50*amp_eb[i]*np.sin(phase_eb[i])
            ya = 50*amp_eb[i]*np.cos(phase_eb[i])
            plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color=[0.1,0.1,0.1])
            
            xa = 50*amp[i]*np.sin(phase[i])
            ya = 50*amp[i]*np.cos(phase[i])
            plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    def cond_polar(self,cond,phase,amp):
        plt.figure()
        instrip = self.ft2['instrip'].to_numpy()
        if cond=='entry_exit':
            
            dstrip = np.diff(instrip)
            dxe = np.where(dstrip>0)[0]+1 
            dxex = np.where(dstrip<0)[0]+1 
        for i in dxe:
            xa = amp[i]*np.sin(phase[i])
            ya = amp[i]*np.cos(phase[i])
            plt.arrow(0,0,xa,ya,length_includes_head=True,head_width=0.005,color=[1,0.3,0.3],alpha=0.25)
        for i in dxex:
            xa = amp[i]*np.sin(phase[i])
            ya = amp[i]*np.cos(phase[i])
            plt.arrow(0,0,xa,ya,length_includes_head=True,head_width=0.005,color=[0.3,0.3,1],alpha=0.25)
            
        xall = amp[dxe]*np.sin(phase[dxe])
        xm = xall.mean()
        yall = amp[dxe]*np.cos(phase[dxe])
        ym = yall.mean()
        plt.arrow(0,0,xm,ym,length_includes_head=True,head_width=0.005,color=[1,0.1,0.1],alpha=1)
        
        xall = amp[dxex]*np.sin(phase[dxex])
        xm = xall.mean()
        yall = amp[dxex]*np.cos(phase[dxex])
        ym = yall.mean()
        plt.arrow(0,0,xm,ym,length_includes_head=True,head_width=0.005,color=[0.3,0.3,1],alpha=1)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()  
            
    def mean_in_plume(self):
        plt.figure()
        eb_phase = self.pdat['offset_eb_phase']
        fsb_phase = self.pdat['offset_fsb_phase']
        #eb_phase = self.phase_eb
        #fsb_phase = self.phase
        strip = self.ft2['instrip'].to_numpy()
        sdiff = np.diff(strip)
        ts = self.pv2['relative_time'].to_numpy()
        tinc = np.mean(np.diff(ts))
        son = np.where(sdiff>0)[0]+1
        soff = np.where(sdiff<0)[0]+1
        son = son[1:]
        soff = soff[1:]
        mn_mat = np.zeros((len(son),20))
        mn_mat2 = np.zeros((len(son),20))
        new_time = np.arange(0,20,dtype=float)
        for i,s in enumerate(son):
            idx_array = np.arange(s-1,soff[i],dtype= int)
            old_time = (idx_array-idx_array[0])
            tp = fsb_phase[idx_array]
            tpi = np.interp(new_time,old_time,tp)
            mn_mat2[i,:] = tpi
            
            tp2 = eb_phase[idx_array]
            tpi_2 = np.interp(new_time,old_time,tp2)
            mn_mat[i,:] = tpi_2
        
        plt_mn = circmean(mn_mat,axis=0,high=np.pi,low=-np.pi)
        std = circstd(mn_mat,axis=0,high=np.pi,low=-np.pi)
        t = new_time
        plt.fill_between(t,plt_mn+std,plt_mn-std,color = [0.6, 0.6, 0.6],zorder=0,alpha = 0.3)
        plt.plot(t,plt_mn,color=[0.3,0.3,0.3],zorder=1)
        
        plt_mn = circmean(mn_mat2,axis=0,high=np.pi,low=-np.pi)
        std = circstd(mn_mat2,axis=0,high=np.pi,low=-np.pi)
        plt.fill_between(t,plt_mn+std,plt_mn-std,color = [0.6, 0.6, 1],zorder=3,alpha = 0.3)
        plt.plot(t,plt_mn,color=[0.3,0.3,0.8],zorder=4)
        plt.xlim([0,19])
        mn = -np.pi
        mx = np.pi
        plt.ylim([-np.pi,np.pi])
        plt.ylabel('Phase')
        plt.yticks([mn,mn/2,0,mx/2,mx],labels=['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'])
        
        plt.plot([0,19],[0,0],color='k',linestyle='--')
        plt.show()
        plt.ylabel('Phase')
        plt.xlabel('In plume time (AU)')
    def polar_movie(self,phase,amp):
        import matplotlib as mpl
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg'
        import networkx as nx

        #mpl.use("TkAgg") 
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.animation import ImageMagickFileWriter, ImageMagickWriter,FFMpegWriter
        # Your specific x and y values
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        instrip = self.ft2['instrip'].to_numpy()
        # Create initial line plot

        fig, ax = plt.subplots(figsize=(10,10))
        line2, = ax.plot([],[],lw=2,color=[0.2,0.2,0.2])
        line, = ax.plot([], [], lw=2,color=[0.2,0.4,1])  # Empty line plot with line width specified
        sc = ax.scatter([],[],color=[0.5,0.5,0.5])

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect('equal')
        # Set axis limits
        #ax.set_xlim(xrange[0], xrange[1])
        

        # Animation update function
        def update(frame):
            # Update the line plot with the current x and y values
            line2.set_data(x[:frame], y[:frame])
            if frame>100:
                line.set_data(x[frame-100:frame], y[frame-100:frame])
            else:
                line.set_data(x[:frame], y[:frame])
            
            if instrip[frame]>0:
                sc.set_offsets(np.column_stack((x[frame],y[frame])))
            
        # Create animation
        anim = mpl.animation.FuncAnimation(fig, update, frames=len(x), interval=0.01)
        plt.show()
    
    def mean_phase_train(self,trng =0.5):
        phase = self.pdat['offset_fsb_upper_phase']
        phase2 = self.pdat['offset_eb_phase']
        #phase2 = self.ft2['ft_heading'].to_numpy()
        tt = self.pv2['relative_time'].to_numpy() 
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        x,y = self.fictrac_repair(x, y)
        
        offs = 0
        tall =np.array([])
        tron,troff,mon,moff,inon,inoff = self.get_train()
        x = x-x[inon[0]]
        # First plume ############################ ###########################
        r = tron[0]
        # First plume
        tson = inon[inon<=r]
        tsoff = inoff[inoff<=r]
        for it,t in enumerate(tsoff[:-1]):
            dx = np.arange(t,tson[it+1])
            tphase = phase[dx]
            tphase2 = phase2[dx]
            trange = tt[dx]
            trange = trange-trange[-1]
            tdx = trange>-trng 
            p = circmean(tphase[tdx],high=np.pi,low=-np.pi)
            plt.scatter(offs,p,color ='k',zorder=10)
            p2 = circmean(tphase2[tdx],high=np.pi,low=-np.pi)
            #plt.plot([offs,offs],[p,p2],color ='k')
            offs = offs+1
            tall = np.append(tall,p)
        side = np.sign(x[dx][-1])
        
        plt.plot([0,offs-1],np.array([0,0])+side*np.pi/2,color='k',linestyle='--')
        
         
        
        old_side= side
        for i,r in enumerate(tron):
            trainon = mon[np.logical_and( mon>r, mon<=troff[i])]
            trainoff = moff[np.logical_and( mon>r, mon<=troff[i])]  
            old_offs = offs
            side = -old_side
            for it,t in enumerate(trainon):
                if it==0:
                    dx = np.arange(t-2,t)
                else:
                    dx = np.arange(trainoff[it-1],t)
                
                tphase = phase[dx]
                tphase2 = phase2[dx]
                trange = tt[dx]
                trange = trange-trange[-1]
                tdx = trange>-trng 
                p = circmean(tphase[tdx],high=np.pi,low=-np.pi)
                plt.scatter(offs,p,color='r',zorder=10)
                p2 = circmean(tphase2[tdx],high=np.pi,low=-np.pi)
                #plt.plot([offs,offs],[p,p2],color ='r')
                tall = np.append(tall,p)
                offs = offs+1
            
            if (i)==(len(tron)-1):
                tson = inon[inon>=troff[i]]
                tsoff = inoff[inoff>=troff[i]]
            else:
                print(i)
                print(len(tron))
                tson = inon[np.logical_and(inon>=troff[i],inon<tron[i+1])]
                tsoff = inoff[np.logical_and(inon>=troff[i],inon<tron[i+1])]
                
           
            
            
            for it,t in enumerate(tsoff[:-1]):
                try:
                    dx = np.arange(t,tson[it+1])
                    tphase = phase[dx]
                    tphase2 = phase2[dx]
                    trange = tt[dx]
                    trange = trange-trange[-1]
                    tdx = trange>-trng 
                    p = circmean(tphase[tdx],high=np.pi,low=-np.pi)
                    plt.scatter(offs,p,color ='k',zorder=10)
                    p2 = circmean(tphase2[tdx],high=np.pi,low=-np.pi)
                    #plt.plot([offs,offs],[p,p2],color ='k')
                    tall = np.append(tall,p)
                    offs = offs+1
                except:
                    x =1
            plt.plot([old_offs,offs-1],np.array([0,0])+side*np.pi/2,color='k',linestyle='--')
            old_side = side
            
        plt.plot(tall,color='k',zorder=-1)     
        plt.plot([0,offs],[0,0],color='k')
       # plt.plot([0,offs],np.array([0,0])+np.pi/2,color='k',linestyle='--')
        #plt.plot([0,offs],np.array([0,0])-np.pi/2,color='k',linestyle='--')
        plt.ylim([-np.pi,np.pi])
        plt.ylabel('mean FC2/EPG phase (deg)')
        plt.yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],labels=[-180,-90,0,90,180])
        plt.xlabel('return/train instance')
        plt.xlim([-0.5,offs])
        
    def plot_train_arrow_mean(self,eb='eb',fsb='fsb_upper',plumeang=22.5,plumewidth=30,bins=100,anum=15,arrowhead=True):
        plt.figure()
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        x,y = self.fictrac_repair(x, y)
        phase = self.pdat['offset_' +fsb+ '_phase'].to_numpy()
        phase_eb = self.pdat['offset_'+eb+'_phase'].to_numpy()
        wd = 0.5*plumewidth*np.cos(np.pi*plumeang/180)
        wd = plumewidth/2+2 # added one for fuzziness of boundary
        paxo = np.array([-wd,wd,wd,-wd])
        tron,troff,mon,moff,inon,inoff = self.get_train()
        offset = 0
        if arrowhead:
            headwidth = 0.3
        else:
            headwidth= 0
        
        for i,r in enumerate(tron):
            tson = inon[inon<=r]
            tsoff = inoff[inoff<=r]
            trajmean = np.zeros((bins,2,len(tsoff)-1))
            phasemean = np.zeros((bins,len(tsoff)-1))
            phasemean_eb = np.zeros((bins,len(tsoff)-1))
            
            trajmean_in = np.zeros((bins,2,len(tsoff)-1))
            phasemean_in = np.zeros((bins,len(tsoff)-1))
            phasemean_in_eb = np.zeros((bins,len(tsoff)-1))
            # Plume 1
            for ti,t in enumerate(tsoff[:-1]):
                
                dx = np.arange(t,tson[ti+1]+2)
                tx = x[dx]
                ty = y[dx]
                tp = phase[dx]
                tpe = phase_eb[dx]
                
                oldtime = np.arange(0,len(tx))
                newtime = np.linspace(0,len(tx),bins)
                txi = np.interp(newtime,oldtime,tx)
                tyi = np.interp(newtime,oldtime,ty)
                subx = txi[0]
                suby = tyi[0]
                txi = txi-subx
                tyi = tyi-suby
                
                tpi = np.interp(newtime,oldtime,fc.unwrap(tp))
                tpi = fc.wrap(tpi)
                
                tpei = np.interp(newtime,oldtime,fc.unwrap(tpe))
                tpei = fc.wrap(tpei)
                
                plt.plot(txi+offset,tyi,color='k',alpha=0.3)
                trajmean[:,0,ti] = txi
                trajmean[:,1,ti] = tyi
                phasemean[:,ti] = tpi
                phasemean_eb[:,ti] = tpei
                
                
                
                dx = np.arange(tson[ti+1]+1,tsoff[ti+1]+1)
                
                tx = x[dx]
                ty = y[dx]
                tp = phase[dx]
                tpe = phase_eb[dx]
                
                oldtime = np.arange(0,len(tx))
                newtime = np.linspace(0,len(tx),bins)
                txi = np.interp(newtime,oldtime,tx)
                tyi = np.interp(newtime,oldtime,ty)
                
                txi = txi-subx
                tyi = tyi-suby
                
                tpi = np.interp(newtime,oldtime,fc.unwrap(tp))
                tpi = fc.wrap(tpi)
                
                tpei = np.interp(newtime,oldtime,fc.unwrap(tpe))
                tpei = fc.wrap(tpei)
                
                plt.plot(txi+offset,tyi,color='r',alpha=0.3)
                trajmean_in[:,0,ti] = txi
                trajmean_in[:,1,ti] = tyi
                phasemean_in_eb[:,ti] = tpei
                
            tmean = np.mean(trajmean,axis=2)
            tmean_in = np.mean(trajmean_in,axis=2)
            tside = np.sign(tmean[-1,0]-tmean[0,0])
            ym = np.max(np.max(trajmean_in[:,1,:]))
            ymin = np.min(np.min(trajmean_in[:,1,:]))
            xdiff = ym*np.tan(tside*np.pi*plumeang/180)
            xdiff2 = ymin*np.tan(tside*np.pi*plumeang/180)
            pax = paxo.copy()
            pax[2:] = pax[2:]+xdiff
            pax[:2] = pax[:2]+xdiff2
            yax = np.array([ymin,ymin,ym,ym])
            plt.fill(pax+offset+tside*wd,yax,color=[0.8,0.8,0.8])
            
            plt.plot(tmean[:,0]+offset,tmean[:,1],color='k')
            pmean = circmean(phasemean,axis=1,high=np.pi,low=-np.pi)
            pmean_eb = circmean(phasemean_eb,axis=1,high=np.pi,low=-np.pi)
            ta = np.linspace(0,bins-1,anum,dtype='int')
            for t in ta:
                
                xa = 3*np.sin(pmean[t])
                ya = 3*np.cos(pmean[t])
                
                plt.arrow(tmean[t,0]+offset,tmean[t,1],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0.3,0.3,1],zorder=10)
                
                xa = 3*np.sin(pmean_eb[t])
                ya = 3*np.cos(pmean_eb[t])
                
                plt.arrow(tmean[t,0]+offset,tmean[t,1],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0,0,0],zorder=10)
            
            plt.plot(tmean_in[:,0]+offset,tmean_in[:,1],color='r')
            pmean = circmean(phasemean_in,axis=1,high=np.pi,low=-np.pi)
            ta = np.linspace(0,bins-1,anum,dtype='int')
            for t in ta:
                
                xa = 3*np.sin(pmean[t])
                ya = 3*np.cos(pmean[t])
                
                plt.arrow(tmean_in[t,0]+offset,tmean_in[t,1],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0.3,0.3,1],zorder=10)
                
                xa = 3*np.sin(pmean_eb[t])
                ya = 3*np.cos(pmean_eb[t])
                
                plt.arrow(tmean_in[t,0]+offset,tmean_in[t,1],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0,0,0],zorder=10)
            
            # Training
            trainon = mon[np.logical_and( mon>r, mon<=troff[i])] # 30s  before
            trainoff = moff[np.logical_and( mon>r, mon<=troff[i])]
            print('Trainoff',trainoff)
            print('trainon',trainon)
            trainoff = np.append(trainon[0]-5,trainoff)
            print('trainoff2',trainoff)
            offset = offset+10
            for ti,t in enumerate(trainoff[:-1]):
                dx = np.arange(t,trainon[ti]+1)
                if len(dx)>50:
                    dx = dx[-50:]
                tx = x[dx]
                ty = y[dx]
                subx = tx[0]
                suby = ty[0]
                tx = tx-subx
                ty = ty-suby
                
                
                tp = phase[dx]
                tp_eb = phase_eb[dx]
                plt.plot(tx+offset,ty,color='k')
                
                
                
                ta = np.linspace(0,len(tp)-1,anum,dtype='int')
                for t in ta:
                    xa = 3*np.sin(tp[t])
                    ya = 3*np.cos(tp[t])
                    
                    plt.arrow(tx[t]+offset,ty[t],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0.3,0.3,1],zorder=10)
                    
                    xa = 3*np.sin(tp_eb[t])
                    ya = 3*np.cos(tp_eb[t])
                    
                    plt.arrow(tx[t]+offset,ty[t],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0,0,0],zorder=10)
                    
                dx = np.arange(trainon[ti],trainoff[ti+1])
                print('to',trainon[ti])
                print('toff',trainoff[ti])
                tx = x[dx]
                ty = y[dx]
                tx = tx-subx
                ty = ty-suby

                tp = phase[dx]
                tp_eb = phase_eb[dx]
                plt.plot(tx+offset,ty,color='r')
                
                
                
                ta = np.linspace(0,len(tp)-1,anum,dtype='int')
                for t in ta:
                    xa = 3*np.sin(tp[t])
                    ya = 3*np.cos(tp[t])
                    
                    plt.arrow(tx[t]+offset,ty[t],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0.3,0.3,1],zorder=10)
                    
                    xa = 3*np.sin(tp_eb[t])
                    ya = 3*np.cos(tp_eb[t])
                    
                    plt.arrow(tx[t]+offset,ty[t],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0,0,0],zorder=10)
                
                
                offset = offset+7.5+np.max(tx)
            
            if (i)==(len(tron)-1):
                tson = inon[inon>=troff[i]]
                tsoff = inoff[inoff>=troff[i]]
            else:
                tson = inon[np.logical_and(inon>=troff[i],inon<tron[i+1])]
                tsoff = inoff[np.logical_and(inon>=troff[i],inon<tron[i+1])]
                
            if len(tsoff)>len(tson):
                tsoff = tsoff[:-1]
                
            # Plume 2
            trajmean = np.zeros((bins,2,len(tsoff)-1))
            phasemean = np.zeros((bins,len(tsoff)-1))
            phasemean_eb = np.zeros((bins,len(tsoff)-1))
            
            trajmean_in = np.zeros((bins,2,len(tsoff)-1))
            phasemean_in = np.zeros((bins,len(tsoff)-1))
            phasemean_in_eb = np.zeros((bins,len(tsoff)-1))
            
            for ti,t in enumerate(tsoff[:-1]):
                
                dx = np.arange(t,tson[ti]+2)
                tx = x[dx]
                ty = y[dx]
                tp = phase[dx]
                tpe = phase_eb[dx]
                
                oldtime = np.arange(0,len(tx))
                newtime = np.linspace(0,len(tx),bins)
                txi = np.interp(newtime,oldtime,tx)
                tyi = np.interp(newtime,oldtime,ty)
                subx = txi[0]
                suby = tyi[0]
                txi = txi-subx
                tyi = tyi-suby
                
                tpi = np.interp(newtime,oldtime,fc.unwrap(tp))
                tpi = fc.wrap(tpi)
                tpei = np.interp(newtime,oldtime,fc.unwrap(tpe))
                tpei = fc.wrap(tpei)
                
                
                plt.plot(txi+offset,tyi,color='k',alpha=0.3)
                trajmean[:,0,ti] = txi
                trajmean[:,1,ti] = tyi
                phasemean[:,ti] = tpi
                phasemean_eb[:,ti] = tpei
                
                dx = np.arange(tson[ti]+1,tsoff[ti+1]+1)
                
                tx = x[dx]
                ty = y[dx]
                tp = phase[dx]
                tpe = phase_eb[dx]
                
                oldtime = np.arange(0,len(tx))
                newtime = np.linspace(0,len(tx),bins)
                txi = np.interp(newtime,oldtime,tx)
                tyi = np.interp(newtime,oldtime,ty)
                
                txi = txi-subx
                tyi = tyi-suby
                
                tpi = np.interp(newtime,oldtime,fc.unwrap(tp))
                tpi = fc.wrap(tpi)
                
                tpei = np.interp(newtime,oldtime,fc.unwrap(tpe))
                tpei = fc.wrap(tpei)
                
                plt.plot(txi+offset,tyi,color='r',alpha=0.3)
                trajmean_in[:,0,ti] = txi
                trajmean_in[:,1,ti] = tyi
                phasemean_in[:,ti] = tpi
                phasemean_in_eb[:,ti] = tpei
                
                
            tmean = np.mean(trajmean,axis=2)
            tmean_in = np.mean(trajmean_in,axis=2)
            tside = np.sign(tmean[-1,0]-tmean[0,0])
            ym = np.max(np.max(trajmean_in[:,1,:]))
            xdiff = ym*np.tan(tside*np.pi*plumeang/180)
            pax = paxo.copy()
            pax[2:] = pax[2:]+xdiff
            
            yax = np.array([0,0,ym,ym])
            plt.fill(pax+offset+tside*wd,yax,color=[0.8,0.8,0.8])
            
            plt.plot(tmean[:,0]+offset,tmean[:,1],color='k')
            pmean = circmean(phasemean,axis=1,high=np.pi,low=-np.pi)
            pmean_eb = circmean(phasemean_eb,axis=1,high=np.pi,low=-np.pi)
            ta = np.linspace(0,bins-1,anum,dtype='int')
            for t in ta:
                
                
                xa = 3*np.sin(pmean[t])
                ya = 3*np.cos(pmean[t])
                
                plt.arrow(tmean[t,0]+offset,tmean[t,1],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0.3,0.3,1],zorder=10)
                
                
                xa = 3*np.sin(pmean_eb[t])
                ya = 3*np.cos(pmean_eb[t])
                
                plt.arrow(tmean[t,0]+offset,tmean[t,1],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0,0,0],zorder=10)
                
            plt.plot(tmean_in[:,0]+offset,tmean_in[:,1],color='r')
            pmean = circmean(phasemean_in,axis=1,high=np.pi,low=-np.pi)
            pmean_eb = circmean(phasemean_in_eb,axis=1,high=np.pi,low=-np.pi)
            ta = np.linspace(0,bins-1,anum,dtype='int')
            for t in ta:
                
                xa = 3*np.sin(pmean[t])
                ya = 3*np.cos(pmean[t])
                
                plt.arrow(tmean_in[t,0]+offset,tmean_in[t,1],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0.3,0.3,1],zorder=10)
                
                xa = 3*np.sin(pmean_eb[t])
                ya = 3*np.cos(pmean_eb[t])
                
                plt.arrow(tmean_in[t,0]+offset,tmean_in[t,1],xa,ya,length_includes_head=arrowhead,head_width=headwidth,color=[0,0,0],zorder=10)
            
            
            
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show() 
            
            
        
    def get_train(self):
        instrip = self.ft2['instrip'].to_numpy()
        mfcon = self.ft2['mfc2_stpt'].to_numpy()
        mfcon = mfcon>0
        mfcint = np.zeros_like(mfcon,dtype='int')
        mfcint[mfcon] = 1
        mfcdiff = np.diff(mfcint)
        mon = np.where(mfcdiff>0)[0]
        moff = np.where(mfcdiff<0)[0]
        
        idiff = np.diff(instrip)
        inon = np.where(idiff>0)[0]
        inoff = np.where(idiff<0)[0]
        
        intr = self.ft2['intrain'].to_numpy().copy()
        
        intr[intr==False] = 0    
        
        intr[intr==True] = 1
        for i,ir in enumerate(intr):
            if np.isnan(ir):
                intr[i] = 0
                
        
        intr[intr>0] =1
        bi,bs = ug.find_blocks(intr,mergeblocks=True,merg_threshold=200)
        for ib,b in enumerate(bi):
            bdx = np.arange(b,b+bs[ib])
            intr[bdx] =1
        
       
        intrdiff = np.diff(intr)
        
        tron = np.where(intrdiff>0)[0]
        troff = np.where(intrdiff<0)[0]
        return tron,troff,mon,moff,inon,inoff
        
    def plot_train_v(self,plumeang=22.5,plumewidth=30,tperiod = 0.5):
        # Set up variables
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        x,y = self.fictrac_repair(x, y)
        tt = self.pv2['relative_time'].to_numpy()
        tt = np.round(tt,decimals=1)
        phase = self.pdat['offset_fsb_upper_phase'].to_numpy()
        amp = self.pdat['amp_fsb_upper']
        instrip = self.ft2['instrip'].to_numpy()
        mfcon = self.ft2['mfc2_stpt'].to_numpy()
        mfcon = mfcon>0
        mfcint = np.zeros_like(mfcon,dtype='int')
        mfcint[mfcon] = 1
        mfcdiff = np.diff(mfcint)
        mon = np.where(mfcdiff>0)[0] #Mass flow controller on
        moff = np.where(mfcdiff<0)[0] # Mass flow controller off
        
        idiff = np.diff(instrip)
        inon = np.where(idiff>0)[0]
        inoff = np.where(idiff<0)[0]
        
        intr = self.ft2['intrain'].to_numpy().copy()
        
        intr[intr==False] = 0    
        
        intr[intr==True] = 1
        for i,ir in enumerate(intr):
            if np.isnan(ir):
                intr[i] = 0
                
        
        intr[intr>0] =1
        bi,bs = ug.find_blocks(intr,mergeblocks=True,merg_threshold=200)
        for ib,b in enumerate(bi):
            bdx = np.arange(b,b+bs[ib])
            intr[bdx] =1
        
       
        intrdiff = np.diff(intr)
        
        tron = np.where(intrdiff>0)[0]
        troff = np.where(intrdiff<0)[0]
        
        
        wd = 0.5*plumewidth*np.cos(np.pi*plumeang/180)
        wd = plumewidth/2+2 # added one for fuzziness of boundary
        pax = np.array([-wd,wd,wd,-wd])
        offset = 0
        
        # Plot V #############################################################
        r = tron[0]
        tson = inon[inon<=r]
        tsoff = inoff[inoff<=r]
        tx = x[tson[0]:tsoff[-1]]
        ty = y[tson[0]:tsoff[-1]]
        t_tt =tt[tson[0]:tsoff[-1]]
        t_tt = t_tt-t_tt[0]
        t_p = phase[tson[0]:tsoff[-1]]
        tis = instrip[tson[0]:tsoff[-1]]
        tside = np.sign(tx[-1]-tx[0])
        tx = tx-tx[0]
        ty = ty-ty[0]
        ym = np.max(ty)
        xdiff = ym*np.tan(tside*np.pi*plumeang/180)
        pax[2:] = pax[2:]+xdiff
        yax = np.array([0,0,ym,ym])
        plt.figure()
        plt.fill(pax+offset,yax,color=[0.8,0.8,0.8])
        
        pax = np.array([-wd,wd,wd,-wd])
        pax[2:] = pax[2:]-xdiff
        plt.fill(pax+offset,yax,color=[0.8,0.8,0.8])
        plt.scatter(tx[tis>0],ty[tis>0],color='r')
        plt.plot(tx+offset,ty,color='k')
        
        for it,t in enumerate(t_tt):
            if np.mod(t,tperiod)==0:
                p = t_p[it]
                xa = 50*amp[it]*np.sin(p)
                ya = 50*amp[it]*np.cos(p)
                plt.arrow(tx[it]+offset,ty[it],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
        
    
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        start_side = np.sign(tx[-1])
            # for loop plot train then plot plume
        plume_off = tsoff[-1]
        for i,r in enumerate(tron):
            #Training #######################################################
            plt.figure()
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            #offset = offset+100
            lastplume= np.max(moff[moff<r])+1
            trainon = mon[np.logical_and( mon>=r, mon<=troff[i])]# -30 #-30s  before
            trainon[0] = max(trainon[0]-30,lastplume)
            #trainon = np.min(np.append(trainon,plume_off))
            
            
            trainoff = moff[np.logical_and( mon>r, mon<=troff[i])]
            trainoff[-1] = min(trainoff[-1],troff[i])
            tx = x[trainon[0]:trainoff[-1]]
            ty = y[trainon[0]:trainoff[-1]]
            
            t_tt =tt[trainon[0]:trainoff[-1]]
            t_tt = t_tt-t_tt[0]
            t_p = phase[trainon[0]:trainoff[-1]]
            
            
            tx = tx-tx[0]
            ty = ty-ty[0]
            tis = mfcint[trainon[0]:trainoff[-1]]
            
            plt.plot(tx+offset,ty,color='k')
            plt.scatter(tx[tis>0]+offset,ty[tis>0],color = 'r')
            
            told = t_tt[0]
            for it,t in enumerate(t_tt):
                if t-told>tperiod:
                    p = t_p[it]
                    xa = 50*amp[it]*np.sin(p)
                    ya = 50*amp[it]*np.cos(p)
                    plt.arrow(tx[it]+offset,ty[it],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
                    told = t
            
            
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            #return
            
            # Plume #########################################################
            #Pick a side
            plt.figure()
            side = start_side*-1
            start_side = side
            
            pon = troff[i]
            if i<len(tron)-1:
                poff = tron[i+1]
            else:
                poff = len(x)
            tx = x[pon:poff]
            ty = y[pon:poff]
            tis = mfcint[pon:poff]
            t_tt = tt[pon:poff]
            t_tt = t_tt-t_tt[0]
            t_p = phase[pon:poff]
            
            tx = tx-tx[0]
            ty = ty-ty[0]
            ty = ty-(1+(plumewidth/2)/np.cos(np.deg2rad(plumeang))) #Annoying off centre parameter
            pax = np.array([-wd,wd,wd,-wd])
            ym = np.max(ty)
            xdiff = ym*np.tan(side*np.pi*plumeang/180)
            xdiffmin = np.min(ty)*np.tan(side*np.pi*plumeang/180)
            pax[:2] = pax[:2]+xdiffmin
            pax[2:] = pax[2:]+xdiff
            yax = np.array([np.min(ty),np.min(ty),np.max(ty),np.max(ty)])
            plt.fill(pax+offset*side,yax,color=[0.8,0.8,0.8]) # check experiment code to get this right
            plt.plot(tx+offset,ty,color='k')
            plt.scatter(tx[tis>0]+offset,ty[tis>0],color = 'r')
            
            
            told = t_tt[0]
            for it,t in enumerate(t_tt):
                if t-told>tperiod:
                    p = t_p[it]
                    xa = 50*amp[it]*np.sin(p)
                    ya = 50*amp[it]*np.cos(p)
                    plt.arrow(tx[it]+offset,ty[it],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
                    told = t
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            
    
    def plot_train_arrows(self,plumeang=22.5,plumewidth=30,tperiod = 0.5):
        
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        x,y = self.fictrac_repair(x, y)
        tt = self.pv2['relative_time'].to_numpy()
        tt = np.round(tt,decimals=1)
        phase = self.pdat['offset_fsb_upper_phase'].to_numpy()
        amp = self.pdat['amp_fsb_upper']
        instrip = self.ft2['instrip'].to_numpy()
        mfcon = self.ft2['mfc2_stpt'].to_numpy()
        mfcon = mfcon>0
        mfcint = np.zeros_like(mfcon,dtype='int')
        mfcint[mfcon] = 1
        mfcdiff = np.diff(mfcint)
        mon = np.where(mfcdiff>0)[0]
        moff = np.where(mfcdiff<0)[0]
        
        idiff = np.diff(instrip)
        inon = np.where(idiff>0)[0]
        inoff = np.where(idiff<0)[0]
        
        intr = self.ft2['intrain'].to_numpy().copy()
        
        intr[intr==False] = 0    
        
        intr[intr==True] = 1
        for i,ir in enumerate(intr):
            if np.isnan(ir):
                intr[i] = 0
                
        
        intr[intr>0] =1
        bi,bs = ug.find_blocks(intr,mergeblocks=True,merg_threshold=200)
        for ib,b in enumerate(bi):
            bdx = np.arange(b,b+bs[ib])
            intr[bdx] =1
        
       
        intrdiff = np.diff(intr)
        
        tron = np.where(intrdiff>0)[0]
        troff = np.where(intrdiff<0)[0]
        
        
        
        
        
        
        wd = 0.5*plumewidth*np.cos(np.pi*plumeang/180)
        wd = plumewidth/2+2 # added one for fuzziness of boundary
        pax = np.array([-wd,wd,wd,-wd])
        offset = 0
        for i,r in enumerate(tron):
            
            # First plume
            
            try:
                
                tson = inon[inon<=r]
                tsoff = inoff[inoff<=r]
                tx = x[tson[0]:tsoff[-1]]
                ty = y[tson[0]:tsoff[-1]]
                t_tt =tt[tson[0]:tsoff[-1]]
                t_tt = t_tt-t_tt[0]
                t_p = phase[tson[0]:tsoff[-1]]
                tis = instrip[tson[0]:tsoff[-1]]
                tside = np.sign(tx[-1]-tx[0])
                tx = tx-tx[0]
                ty = ty-ty[0]
                ym = np.max(ty)
                xdiff = ym*np.tan(tside*np.pi*plumeang/180)
                pax[2:] = pax[2:]+xdiff
                yax = np.array([0,0,ym,ym])
                plt.figure()
                plt.fill(pax+offset,yax,color=[0.8,0.8,0.8])
                plt.scatter(tx[tis>0],ty[tis>0],color='r')
                plt.plot(tx+offset,ty,color='k')
                
                for it,t in enumerate(t_tt):
                    if np.mod(t,tperiod)==0:
                        p = t_p[it]
                        xa = 50*amp[it]*np.sin(p)
                        ya = 50*amp[it]*np.cos(p)
                        plt.arrow(tx[it]+offset,ty[it],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
            except:
                print('No first plume')
            
            
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            
            #Training
            plt.figure()
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            offset = offset+100
            trainon = mon[np.logical_and( mon>=r, mon<=troff[i])]-300 # 30s  before
            
            trainoff = moff[np.logical_and( mon>r, mon<=troff[i])]
            tx = x[trainon[0]:trainoff[-1]]
            ty = y[trainon[0]:trainoff[-1]]
            
            t_tt =tt[trainon[0]:trainoff[-1]]
            t_tt = t_tt-t_tt[0]
            t_p = phase[trainon[0]:trainoff[-1]]
            
            
            tx = tx-tx[0]
            ty = ty-ty[0]
            tis = mfcint[trainon[0]:trainoff[-1]]
            
            plt.plot(tx+offset,ty,color='k')
            plt.scatter(tx[tis>0]+offset,ty[tis>0],color = 'r')
            
            told = t_tt[0]
            for it,t in enumerate(t_tt):
                if t-told>tperiod:
                    p = t_p[it]
                    xa = 50*amp[it]*np.sin(p)
                    ya = 50*amp[it]*np.cos(p)
                    plt.arrow(tx[it]+offset,ty[it],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
                    told = t
            
            
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            
            
            # Second plume
            # plt.figure()
            # ax = plt.gca()
            # ax.set_aspect('equal', adjustable='box')
            # pax = np.array([-wd,wd,wd,-wd])
            # offset = offset+100
          
            # if (i)==(len(tron)-1):
            #     tson = inon[inon>=troff[i]]
            #     tsoff = inoff[inoff>=troff[i]]
                
            #    # tsoff = inoff[]
                   
            # else:
            #     tson = inon[np.logical_and(inon>=troff[i],inon<tron[i+1])]
            #     tsoff = inoff[np.logical_and(inon>=troff[i],inon<tron[i+1])]
            
            # print('inon',inon)
            # print('tson',tson)
            # print('inoff',inoff)
            # print('tsoff',tsoff)
            
            # tx = x[tson[0]:tsoff[-1]]
            # ty = y[tson[0]:tsoff[-1]]
            # tis = instrip[tson[0]:tsoff[-1]]
            # tside = np.sign(tx[-1]-tx[0])
            # t_tt =tt[tson[0]:tsoff[-1]]
            # t_tt = t_tt-t_tt[0]
            # t_p = phase[tson[0]:tsoff[-1]]
            # print(tside)
            # tx = tx-tx[0]
            # ty = ty-ty[0]
            # ym = np.max(ty)
            # xdiff = ym*np.tan(tside*np.pi*plumeang/180)
            # print(xdiff)
            # pax = pax+wd*tside
            # pax[2:] = pax[2:]+xdiff
            # yax = np.array([0,0,ym,ym])
            # plt.fill(pax+offset,yax,color=[0.8,0.8,0.8])
            # #plt.scatter(tx[tis>0],ty[tis>0],color='r')
            # plt.plot(tx+offset,ty,color='k')
            
            # told = t_tt[0]   
            # for it,t in enumerate(t_tt):
                
            #     if t-told>tperiod:
            #         p = t_p[it]
            #         xa = 50*amp[it]*np.sin(p)
            #         ya = 50*amp[it]*np.cos(p)
            #         plt.arrow(tx[it]+offset,ty[it],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
            #         told = t
            
            
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()    
        
        
    def fictrac_repair(self,x,y):
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
    def bumpstraighten(self,x,y,heading):
        
        ft2 = self.ft2
        ft = self.ft
        obumps = ft['bump'].to_numpy()
        obumps_u = obumps[np.abs(obumps)>0]
        obumpsfr = ft['frame'][np.abs(obumps)>0]
        bumps = ft2['bump']
        frames = ft2['frame']
        bumps_new = np.zeros_like(bumps)
        for i,f in enumerate(obumpsfr):
            
            frd = frames-f
            w = np.argmin(np.abs(frd))
            
            bumps_new[w] = obumps_u[i]
        
        
        bumps = bumps_new
        binst = np.where(np.abs(bumps)>0)[0]
        xnew = x.copy()
        ynew = y.copy()
        headingnew = heading.copy()
        tbold = 0
        for b in range(len(binst)-1):
            bi = binst[b]
            tb = bumps[bi]+tbold
            tbold = tb
            bdx = np.arange(bi,binst[b+1],step=1,dtype=int)
            bc =np.cos(-tb)
            bs = np.sin(-tb)
            tx = x[bdx]
            ty = y[bdx]
            tx = tx-tx[0]
            ty = ty-ty[0]
            tx2 = tx*bc-ty*bs
            ty2 = tx*bs+ty*bc
            dx = tx2[0]-xnew[bdx[0]-1]
            dy = tx2[0]-ynew[bdx[0]-1]
            tx2 = tx2-dx
            ty2 = ty2-dy
            xnew[bdx] = tx2
            ynew[bdx] = ty2
            
            th = heading[bdx]+tb
            tc = np.cos(th)
            ts = np.sin(th)
            th = np.arctan2(ts,tc)
            headingnew[bdx] = th
            
            
        dx = xnew[(bdx[-1]+1)]-xnew[bdx[-1]]
        xnew[(bdx[-1]+1):] = xnew[(bdx[-1]+1):]-dx
        dy = ynew[(bdx[-1]+1)]-ynew[bdx[-1]]
        ynew[(bdx[-1]+1):] = ynew[(bdx[-1]+1):]-dy
        
        return xnew,ynew,headingnew
        
                
        
    