# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:02:48 2024

@author: dowel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utilities.utils_general import utils_general as ug
#%%
class opto:
    def __init__(self):
        self.name = 'opto'
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        idx =idx
        return idx
    
    def opto_raster(self,meta_data,df,offset=0,distance_thresh=2000):
        pon = pd.Series.to_numpy(df['instrip'])
        stimon = np.where(pon>0)[0][0]
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        x,y = self.fictrac_repair(x,y)
        y = y-y[stimon]
        ym = np.max(y[pon>0])
        ym = np.round(ym)
        led = df['led1_stpt'].to_numpy()
        s_type = meta_data['stim_type']
        ls = np.where(led==0)[0][0]
        tt = ug.get_ft_time(df)
        tt = tt-tt[ls]
        plt.fill([tt[0],tt[-1],tt[-1],tt[0]],[offset,offset,offset+1,offset+1],color=[0.7,0.7,0.7],linewidth=0)
        
        #plt.fill_between(tt,pon*0+offset,pon*0+offset+1,color=[0.7,0.7,0.7])
        
        bstart,bsize = ug.find_blocks(pon)
        for ib,b in enumerate(bstart):
            plt.fill([tt[b],tt[b+bsize[ib]-1],tt[b+bsize[ib]-1],tt[b]],[offset,offset,offset+1,offset+1],color='r',linewidth=0)
        
        
        #plt.fill_between(tt,pon+offset,pon*0+offset,color='r',linewidth=0)
        if ym>=distance_thresh:
            plt.text(tt[-1]+50,offset+0.5,str(ym),color='r')
        else:
            plt.text(tt[-1]+50,offset+0.5,str(ym),color='k')
    def opto_return_raster(self,meta_data,df,offset=0):
        pon = pd.Series.to_numpy(df['instrip'])
        lon = df['led1_stpt'].to_numpy()<1
        lon1 = np.where(lon)[0][0]
        stimon = np.where(pon>0)[0][0]
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        x,y = self.fictrac_repair(x,y)
        y = y-y[stimon]
        dx = np.diff(x)
        dy = np.diff(y)
        dd = np.sqrt(dx**2+dy**2)
        distance = np.cumsum(dd)
        
        t = self.get_time(df)
        dt = np.mean(np.diff(t))
        ledon = df['led1_stpt'].to_numpy()<1
        msize = np.round(0.5/dt)
        
        bstart,bsize = ug.find_blocks(pon,mergeblocks=True,merg_threshold=msize)
        tdists = np.zeros(len(bstart)-1)
        for i,b in enumerate(bstart[:-1]):
            bdx = np.arange(b+bsize[i],bstart[i+1],dtype='int')
            tdist = distance[bdx]
            tdists[i] = tdist[-1]-tdist[0]
        x =np.arange(0,len(tdists))
        try:
            xmid = np.where((bstart[:-1]+bsize[:-1])>lon1)[0][0]
        except:
            xmid = len(tdists)
        print(xmid)
        plt.plot(x-xmid,tdists+offset,color='k')
        plt.scatter(x[(bstart[:-1]+bsize[:-1])<lon1]-xmid,tdists[(bstart[:-1]+bsize[:-1])<lon1]+offset,color='k',zorder=2)
        plt.scatter(x[(bstart[:-1]+bsize[:-1])>lon1]-xmid,tdists[(bstart[:-1]+bsize[:-1])>lon1]+offset,color='r',zorder=2)
    def plot_first_stim(self,df,xoffset=0):
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        x,y = self.fictrac_repair(x,y)
        pon = pd.Series.to_numpy(df['instrip']>0)
        lon = df['led1_stpt'].to_numpy()<1
        #firstlon = np.where(lon)[0][0]
        londons,lonsize = ug.find_blocks(lon,mergeblocks=True,merg_threshold=10)
        firstlon = londons[0]
        pw = np.where(pon)
        x = x-x[pw[0][0]]
        y = y-y[pw[0][0]]
        x = x+xoffset
        
        t = self.get_time(df)
        dt = np.mean(np.diff(t))
        msize = np.round(0.5/dt)
        instrip = df['instrip']
        bstart,bsize = ug.find_blocks(instrip,mergeblocks=True,merg_threshold=msize)
        bstart = bstart[bsize>msize]
        bsize = bsize[bsize>msize]
        #bstart,bsize = ug.find_blocks(pon)
        bs1 = np.max(bstart[bstart<firstlon])
        try:
            bs2 = np.min(bstart[bstart>firstlon])
        except:
            bs2 = len(x)
        bdx = np.arange(bs1,bs2)
        ty = y[bdx]
        tx = x[bdx]
        tymn = np.min(ty)
        tymx = np.max(ty)
        ythresh = y[firstlon]
        plt.fill(np.array([-25,25,25,-25])+xoffset,[tymn,tymn,tymx,tymx],color=[0.7,0.7,0.7])
        plt.plot(tx[ty<=ythresh],ty[ty<=ythresh],color='k')
        plt.plot(tx[ty>=ythresh],ty[ty>=ythresh],color=[1,0.8,0.8])
        plt.gca().set_aspect('equal')
        plt.show()
        
        
        
    def plot_plume(self,meta_data,df):
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
            xsub = self.find_nearest(y,yplus)
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
    def plot_traj_scatter(self,df):
        # Simple plot to show tajectory and scatter of when experiencing odour 
        # and LED
        # Useful for when there is not much information about the experiment
        plt.figure(figsize=(16,16))
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        led_on = df['led1_stpt']==0
        in_s = df['instrip']
        x,y = self.fictrac_repair(x,y)
        plt.plot(x,y,color = 'k')
        plt.scatter(x[in_s],y[in_s],color=[0.5, 0.5, 0.5])
        plt.scatter(x[led_on],y[led_on],color= [1,0.5,0.5],marker='+')
        plt.gca().set_aspect('equal')
        plt.show()
        
        
    def plot_plume_simple(self,meta_data,df):
        
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        led_on = df['led1_stpt'].to_numpy()==0
        instrip = df['instrip'].to_numpy()
        a_s = meta_data['act_inhib']
        #x,y = self.fictrac_repair(x,y)
        s_type = meta_data['stim_type']
        #plt.figure(figsize=(16,16))
        if a_s=='act':
            led_colour = [1,0.8,0.8]
        elif a_s=='inhib':
            led_colour = [0.7, 1, 0.7]
            
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

            #xmplume = yrange[1]/np.tan(pi*(pa/180))
            
            if pa ==90:
                xp = [xrange[0], xrange[1],xrange[1], xrange[0]]
                yp = [psize/2, psize/2,-psize/2,-psize/2]
                xo = [-xlm,xlm,xlm,-xlm, -xlm ]
            else :
                xp = [-psize/2, yrange[1]*np.tan(pi*(pa/180))-psize/2,yrange[1]*np.tan(pi*(pa/180))+psize/2, psize/2,-psize/2]
                yp = [0, yrange[1], yrange[1],0,0]
                xo = [-xlm,xlm,xlm,-xlm, -xlm ]
            #pan = meta_data['PlumeAngle']

            if meta_data['ledOny']=='all':
                lo = yrange[0]
            else:
                lo = meta_data['ledOny']
            
            if meta_data['ledOffy']=='all':
                loff = yrange[1]
                
            else:
                loff = meta_data['ledOffy']
            yo = [lo,lo,loff,loff,lo]
            if meta_data['LEDoutplume']:
                plt.fill(xo,yo,color = led_colour,alpha =0.5)
            
            if loff<yrange[1]:
                while loff<yrange[1]:
                    loff = loff+1000
                    lo = lo+1000
                    yo = [lo,lo,loff,loff,lo]
                    plt.fill(xo,yo,color = led_colour,alpha=0.5)
            
            plt.fill(xp,yp,color =[0.8,0.8,0.8])
            # Add in extra for repeated trials
            plt.plot(x[pw[0][0]:],y[pw[0][0]:],color='k')
            plt.plot(x[0:pw[0][0]],y[0:pw[0][0]],color=[0.5,0.5,0.5])
            if meta_data['LEDinplume']:
                plt.fill(xo,yo,color = led_colour,alpha= 0.5)
            
                plt.scatter(x[led_on],y[led_on], color = [0.8, 0.8 ,0.2])
            #yxlm = np.max(np.abs(np.append(yrange,xrange)))
            #ymn = np.mean(yrange)
            plt.ylim([np.min(y),np.max(y)])
            #plt.ylim([ymn-(yxlm/2), ymn+(yxlm/2)])
            #plt.xlim([-1*(yxlm/2), yxlm/2])
        elif s_type == 'pulse':
            led = df['led1_stpt']==0
            plt.scatter(x[led],y[led],color=led_colour)
            
            plt.plot(x,y,color='k')
        elif s_type == 'alternation':
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
            #plt.scatter(x[led_on],y[led_on],color='r')
            if pa ==90:
                xp = [xrange[0], xrange[1],xrange[1], xrange[0]]
                yp = [psize/2, psize/2,-psize/2,-psize/2]
                xo = [-xlm,xlm,xlm,-xlm, -xlm ]
            else :
                xp = [-psize/2, yrange[1]*np.tan(pi*(pa/180))-psize/2,yrange[1]*np.tan(pi*(pa/180))+psize/2, psize/2,-psize/2]
                yp = [0, yrange[1], yrange[1],0,0]
                xo = [-xlm,xlm,xlm,-xlm, -xlm ]
            plt.fill(xp,yp,color =[0.8,0.8,0.8],zorder=0)
            led_diff = np.diff(led_on.astype(int))

            lon = np.where(led_diff>0)[0]
            loff = np.where(led_diff<0)[0]
            #print(lon)
            plt.plot([-100,100],[meta_data['ledOny'],meta_data['ledOny']],color='k',linestyle='--')
            try:
                plt.plot(x[0:lon[0]],y[0:lon[0]],color='k')
                if len(lon)>len(loff):
                    plt.plot(x[lon[-1]:],y[lon[-1]:],color=led_colour)
                    lon = lon[:-1]
                else:
                    plt.plot(x[loff[-1]:],y[loff[-1]:],color='k')
                    
                for il,l in enumerate(lon):
                    plt.plot(x[l:loff[il]],y[l:loff[il]],color=led_colour)
                for il,l in enumerate(loff[:-1]):
                    print(l)
                    xsmall = x[loff[il]+1:lon[il+1]-2]
                    ysmall = y[loff[il]+1:lon[il+1]-2]
                    
                    plt.plot(xsmall,ysmall,color='k')
                    
                    inplume = instrip[loff[il]+1:lon[il+1]-2]
                    
                    #plt.plot(xsmall[inplume],ysmall[inplume],color='k')
                    #plt.plot(xsmall[inplume==False],ysmall[inplume==False],color=[0.5,0.5,1])
            except:
                plt.plot(x,y,color='k')
        elif s_type=='alternation_jump':
            ac = df['adapted_center'].to_numpy()
            psize =meta_data['PlumeWidth']
            pon = pd.Series.to_numpy(df['instrip']>0)
            pw = np.where(pon)[0]
            
            x = x-x[pw[0]]
            y = y-y[pw[0]]
            ac[np.isnan(ac)] = 0
            d_ac = np.diff(ac)
            jumps = np.where(np.abs(d_ac)>0)[0]+1
            # Plot first plume
            xp = np.array([-psize/2,-psize/2,psize/2,psize/2])
           
            jplm = np.max(pw[pw<jumps[0]])
            ym = y[jplm]
            yp = np.array([0,ym,ym,0])
            plt.fill(xp,yp,color=[0.5,0.5,0.5],alpha=0.5)
            for i,j in enumerate(jumps):
                jplm = np.max(pw[pw<j])
                
                tj = ac[j]
                xpj = xp+tj
                ymin = y[jplm]
                if i<len(jumps)-1:
                    ymax =y[np.max(pw[pw<jumps[i+1]])]
                else:
                    ymax = np.max(y)
                yp = np.array([ymin,ymax,ymax,ymin])
                plt.fill(xpj,yp,color=[0.5,0.5,0.5],alpha=0.5)
                
            #plt.scatter(x[pon],y[pon],color='r')    
            plt.scatter(x[led_on],y[led_on],color=led_colour)                     
            plt.plot(x,y,color='k')
            
        plt.gca().set_aspect('equal')
        plt.show()
        
    def plot_plume_horizontal(self,meta_data,df):
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        
        
        x,y = self.fictrac_repair(x,y)
        
        pon = pd.Series.to_numpy(df['instrip']>0)
        pw = np.where(pon)
        x = x-x[pw[0][0]]
        y = y-y[pw[0][0]]
        
        pa = meta_data['PlumeWidth']
        plt.figure(figsize=(16,16))
        yrange = [min(y), max(y)]
        xrange = [min(x), max(x)]
        x_plm = [xrange[0], xrange[0], xrange[1],xrange[1]]
        y_plm = [-pa/2, pa/2, pa/2, -pa/2]
        
        plt.fill(x_plm,y_plm,color =[0.8,0.8,0.8])
        x_on = meta_data['ledOnx']
        x_off = meta_data['ledOffx']
        y_on = meta_data['ledOny']
        y_off= meta_data['ledOffy']
        y_stm = [y_on, y_off, y_off,y_on]
        rep_int = meta_data['RepeatInterval']
        
        a_s = meta_data['act_inhib']
        if a_s=='act':
            led_colour = [1,0.8,0.8]
        elif a_s=='inhib':
            led_colour = [0.8, 1, 0.8]
        if xrange[0]<0:
            
            xr = -np.arange(0,np.abs(xrange[0]),rep_int)
            print(xr)
            for i in xr:
                
                x_stm = [i-x_on, i-x_on,i-x_off,i-x_off]
                plt.fill(x_stm,y_stm,color=led_colour)
        if xrange[1]>0:
            
            xr = np.arange(0,np.abs(xrange[1]),rep_int)
            print(xr)
            for i in xr:
                
                x_stm = [i+x_on, i+x_on,i+x_off,i+x_off]
                plt.fill(x_stm,y_stm,color=led_colour)
        led_on = df['led1_stpt']<1
        plt.scatter(x[led_on],y[led_on],color='r')    
        plt.plot(x,y,color='k')
        
        
        plt.gca().set_aspect('equal')
        plt.show()
    
        
    def light_pulse_pre_post(self,meta_data,df):
        plt.figure(figsize=(10,10))
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        x,y = self.fictrac_repair(x,y)
        t = self.get_time(df)
        
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
    def get_time(self,df):
        t = pd.Series.to_numpy(df['timestamp'])
        t = np.array(t,dtype='str')
        t_real = np.empty(len(t),dtype=float)
        for i,it in enumerate(t):
            tspl = it.split('T')
            tspl2 = tspl[1].split(':')
            t_real[i] = float(tspl2[0])*3600+float(tspl2[1])*60+float(tspl2[2])
        t_real = t_real-t_real[0]
        return t_real
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
    def extract_stats(self,meta_data,df):
        
        # Stats to get: med max dist from plume, med time outside plume, length traj outside,
        # med vel outside
        plume_an =  np.pi*meta_data['PlumeAngle']/180
        edge_vec = np.array([np.cos(plume_an),np.sin(plume_an)])
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        x,y = self.fictrac_repair(x,y)
        t = self.get_time(df)
        led = df['led1_stpt']

        strip_on = pd.Series.to_numpy(df['strip_thresh'])
        strip_on =np.isnan(strip_on)
        st_on = np.where(~strip_on)[0][0]
        instrip = pd.Series.to_numpy(df['instrip'])
        adapt_cent = pd.Series.to_numpy(df['adapted_center'])
        l_edge = y*np.sin((plume_an))-instrip+adapt_cent
        r_edge = y*np.sin((plume_an))+instrip+adapt_cent
        # Ignore pre period for now
        x = x[st_on:]
        y = y[st_on:]
        t = t[st_on:]
        l_edge = l_edge[st_on:]
        r_edge = r_edge[st_on:]
        instrip = instrip[st_on:].astype(int)
        led = led[st_on:]
        exits = np.where(np.diff(instrip)==-1)[0]+1
        entries = np.where(np.diff(instrip)==1)[0]+1
        
        if (len(exits)-len(entries))==1:
            exits = exits[:-1]
        
        # remove first entry and exit as will not be representative of tracking:
        # May want to make this larger in future iterations
        t_ex = exits[1:]
        t_ent = entries[1:]
        data_array = np.empty([len(t_ex),5])    
        for i,ex in enumerate(t_ex):
            en = t_ent[i]
            ys = y[ex:en]
            xs = x[ex:en]
            ts = t[ex:en]
            tled = led[ex:en]
            xz = xs-xs[0]
            yz = ys-ys[0]
            tz = ts-ts[0]
            xz = xz[:, np.newaxis]
            yz = yz[:, np.newaxis]
            xy = np.append(xz,yz,axis=1)
            edist = np.matmul(xy,edge_vec)
            ledsum = np.sum(tled)
            
            if ledsum==len(tled):
                data_array[i,0] = 0
            elif ledsum==0:
                data_array[i,0] =1 
            else :
                data_array[i,0] = 2
                
            # Max distance from plume: check for slanted/jumping plumes    
            data_array[i,1] = np.max(edist)
            # Time out of plume
            data_array[i,2] = ts[-1]-ts[0]
            # Path length
            dxy = np.diff(xy,axis =0)
            mxy = np.sqrt(np.sum(dxy**2,axis=1))
            data_array[i,3] = np.sum(mxy)
            # Median speed
            data_array[i,4] = np.median(mxy)
            
        min_time = 0.5 
        keep = data_array[:,2]>min_time
        data_array = data_array[keep,:]
        mdn_data = np.empty([3,5],dtype=float)
        mn_data = np.empty([3,5],dtype=float)
        for i in range(3):
            dx = data_array[:,0].astype(int)==i
            mn_data[i,:] = np.mean(data_array[dx,:],axis=0)
            mdn_data[i,:] = np.median(data_array[dx,:],axis=0)
        out_dict = {'all data': data_array,
                    'mean data': mn_data,
                    'median data': mdn_data
            }
        return out_dict
    def get_ret_efficiency(self,df,tilt=0,min_duration=0.5):
        """Gets efficiencies of each plume return"""
        jumps = False
        if 'left_border'in df.columns:
            jumps = True
        
        e_e = self.get_entries_exits(df)
        # Get each entry and exit
        x = df['ft_posx'].to_numpy()
        y = df['ft_posy'].to_numpy()
        x,y = self.fictrac_repair(x,y)
        
        xy = np.append(x[:,np.newaxis],y[:,np.newaxis],axis=1)
        if tilt>0 and tilt<90:
            tsign = -np.sign(df['adapted_center'][e_e[0,0]:e_e[0,1]].mean()) # adapted center for first entry, animal should always go upwind for this
        else:
            tsign = 1
            
        tilt = np.pi*(tilt/180)
        proj_vec = np.array([np.cos(tilt)*tsign,np.sin(tilt)])
        
        
        tt =ug.get_ft_time(df)
        dt = np.mean(np.diff(tt))
        minlen = int(min_duration/dt)
        #  Efficiency = perp distance/pathlength
        
        de = e_e[:,2]-e_e[:,1]
        e_e = e_e[de>=minlen,:]
        output = np.zeros(len(e_e))
        
        for i,e in enumerate(e_e):
            dx = np.arange(e[1],e[2])
            txy = xy[dx,:]
            txy = txy-txy[0,:]
            tproj = np.matmul(txy,proj_vec.T)
            pmax = np.max(tproj)-np.min(tproj)
            # if jumps:
            #     dj = np.abs(df['left_border'][e[0]+2]-df['left_border'][e[2]-1])
            #     pmax = pmax+dj
            
            dtxy = np.diff(txy,axis=0)
            ddist = np.sqrt(np.sum(dtxy**2,axis=1))
            pathlen = np.sum(ddist)
            output[i] = pmax/pathlen
            
            
            
            # if output[i]>0.5:
            #     print(output[i])
            #     print(minlen)
            #     print(e[2]-e[1])
            #     plt.figure()
            #     plt.subplot(1,2,1)
            #     plt.plot(tproj)
            #     plt.plot(txy[:,0],color='k')
            #     plt.plot(txy[:,1],color='r')
            #     plt.subplot(1,2,2)
            #     plt.plot(txy[:,0],txy[:,1],color='k')
            #     g=plt.gca()
            #     g.set_aspect('equal')
            #     raise ValueError('Not correct efficiency')
    
        return output
            
            
            
            
    def get_entries_exits(self,df,ent_duration=.25):
        
        try:
            ins = df['instrip'].to_numpy()
        except:
            ins = df['mfc2_stpt'].to_numpy()>0
            if np.sum(ins)==0:
                ins = df['mfc3_stpt'].to_numpy()>0
                if np.sum(ins)==0:
                    print('Cannot detect odour onset')
                    return
            ins = ins.astype(int)
        tt = ug.get_ft_time(df)
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
    def extract_stats_alternation(self,meta_data,df):
        """
        
        Parameters
        ----------
        meta_data : TYPE
            DESCRIPTION.
        df : TYPE
            DESCRIPTION.

        Returns out_dict
        -------
        Function will return datapoints to do stats on from alternation inhibition
        experiments. These will be: path length, x distance from plume, time outside
        plume for inhib and excitation epochs. The data will be the raw values,
        ratios and medians/means of raw values and ratios.

        """
        
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        x,y = self.fictrac_repair(x,y)
        t = self.get_time(df)
        led = df['led1_stpt']
        led_on = led==0 
        led_diff = np.diff(led_on.astype(int))
        instrip = df['instrip']
        lon = np.where(led_diff>0)[0]
        loff = np.where(led_diff<0)[0]
        isd = np.diff(instrip)
        
        
        # Clip data from here
        if len(lon)>len(loff):
            lon = lon[:-1]
        
        data_lon = np.empty((len(lon),4))
        datakeys = ['max_x','y_travelled','displacement','time_outside']
        # Go through led on epochs
        for i,il in enumerate(lon):
            
            dx = np.arange(il,loff[i],dtype=int)
            tx = x[dx]
            ty = y[dx]
            tx = tx-tx[0]
            ty = ty-ty[0]
            data_lon[i,0] = np.max(np.abs(tx))
            data_lon[i,1] = ty[-1]
            dtx = np.diff(tx)
            dty = np.diff(ty)
            sdtx = np.sum(np.abs(dtx))
            sdty = np.sum(np.abs(dty))
            data_lon[i,2] = np.sqrt(sdtx**2+sdty**2)
            tt = t[dx]
            tt = tt-tt[0]
            data_lon[i,3] = tt[-1]
            if data_lon[i,3]==0:
                print(il,loff[i])
        # Go through led off epochs
        
        
        
        data_loff = np.empty((len(loff)-1,4)) #1 part shorter because of we are taking data from last led,
        
        
        for i, il in enumerate(loff[:-1]):
            dx = np.arange(il,lon[i+1],dtype=int)
            t_is = instrip[dx]
            tx = x[dx]
            ty = y[dx]
            tt = t[dx]
            tt = tt[t_is==0]
            tx = tx[t_is==0]
            ty = ty[t_is==0]
            tx = tx-tx[0]
            ty = ty-ty[0]
            tt = tt-tt[0]
            
            data_loff[i,0] = np.max(np.abs(tx))
            data_loff[i,1] = ty[-1]
            dtx = np.diff(tx)
            dty = np.diff(ty)
            sdtx = np.sum(np.abs(dtx))
            sdty = np.sum(np.abs(dty))
            data_loff[i,2] = np.sqrt(sdtx**2+sdty**2)
            
            data_loff[i,3] = tt[-1]
        
        
        
        ratio_dx = np.logical_and(data_loff[:,3]>0.25,data_lon[:-1,3]>0.5)
        ratio_dx = np.where(ratio_dx)[0]
        
        
        ratios = np.divide(data_lon[ratio_dx,:],data_loff[ratio_dx,:])
        ratmed = np.median(ratios,axis=0)
        ratmean = np.mean(ratios,axis=0)
        ratmed_log = np.median(np.log10(ratios[:,[0,2,3]]),axis=0)
            
        out_dict = {'Column Names': datakeys,
                    'Data_ledON':data_lon,
                    'Data_ledOFF':data_loff,
                    'Ratios':ratios,
                    'Median Ratios': ratmed,
                    'Mean Ratios': ratmean,
                    'Median Ratios Log':ratmed_log,
                    'Ratio_dx':ratio_dx
                    }
        return out_dict
    
    def extract_stats_threshold(self,df,ythresh,statchoice):
        output_data = np.array([])
        x = pd.Series.to_numpy(df['ft_posx'])
        y = pd.Series.to_numpy(df['ft_posy'])
        x,y = self.fictrac_repair(x,y)
        dx = np.diff(x)
        dy = np.diff(y)
        dd = np.sqrt(dx**2+dy**2)
        distance = np.cumsum(dd)
        
        t = self.get_time(df)
        dt = np.mean(np.diff(t))
        ledon = df['led1_stpt'].to_numpy()<1
        msize = np.round(0.5/dt)
        instrip = df['instrip']
        ydx = y>ythresh
        ydxw = np.where(ydx)[0]
        
        bstart,bsize = ug.find_blocks(instrip,mergeblocks=True,merg_threshold=msize)
        bsdx = np.in1d(bstart,ydxw)
        
        bstarty = bstart[bsdx]
        bsizey  = bsize[bsdx]
        
        
        for s in statchoice:
            if s=='mean dist':
                data = np.zeros(len(bstarty[:-1]))
                for i,b in enumerate(bstarty[:-1]):
                    bdx = np.arange(b+bsizey[i],bstarty[i+1],dtype='int')
                    tdist = distance[bdx]
                    data[i] = tdist[-1]-tdist[0]
                
                output_data = np.append(output_data,np.mean(data))
                
            elif s== 'mean ret time':
                data = np.zeros(len(bstarty[:-1]))
                for i,b in enumerate(bstarty[:-1]):
                    bdx = np.arange(b+bsizey[i],bstarty[i+1],dtype='int')
                    tdist = t[bdx]
                    data[i] = tdist[-1]-tdist[0]
                output_data = np.append(output_data,np.mean(data))
                
            elif s== 'returns per m':
                print(len(bstarty))
                if len(bstarty)<2:
                    data = np.nan
                else:
                    runlen = y[bstarty[-1]] - y[bstarty[0]+bsizey[0]]
                    data = len(bstarty)/runlen
                    data = data*1000
                output_data = np.append(output_data,data)
                
            elif s == 'run length':
                if len(bstarty)<1:
                    data = 0
                else:
                    rmax = 2000-ythresh
                    runlen = y[bstarty[-1]+bsizey[-1]-1] - ythresh
                    data = np.min(np.append(runlen,rmax))
                output_data = np.append(output_data,data)
            elif s =='first return':
                lon = np.where(ledon)[0][0]
                lins = instrip[lon]
                
                indices = np.where(bstart > lon)[0]
                if indices.size == 0:
                    print('did not make back')
                    if lins==1:
                        retdx = np.arange(bstart[-1]+bsize[-1],len(x)-1,dtype='int')
                    else:
                        retdx = np.arange(lon,len(x)-1,dtype='int')
                        
                    tdist = distance[retdx]
                    
                    data = tdist[-1]-tdist[0]
                else:
                    next_bstart = np.min(indices)
                    if lins==1:
   
                        retdx = np.arange(bstart[next_bstart-1]+bsize[next_bstart-1],bstart[next_bstart],dtype='int')
                    else:
                        retdx = np.arange(lon,bstart[next_bstart],dtype='int')
                    
                    tdist = distance[retdx]
                    
                    data = tdist[-1]-tdist[0]
                output_data = np.append(output_data,data)
            elif s == 'return status':
                lon = np.where(ledon)[0][0]
                lins = instrip[lon]
                next_bstart = np.where(bstart>lon)[0]
                data = 0
                if lins==1:
                    data = data+1
                    
                if len(next_bstart)<1:
                    data = data+0.1
                    
                #Code: 0.0 stim on outside plume returned after stim, 
                # 1.0 stim on in plume returned after stim
                # 0.1 stim on outside plume left after stim
                # 1.1 stim on inside plume left after stim
                output_data = np.append(output_data,data)
        return output_data
        
        
        
        