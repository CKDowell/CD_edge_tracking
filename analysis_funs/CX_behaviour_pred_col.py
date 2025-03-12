# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:36:02 2025

@author: dowel

Objective:
    1. To use regression methods to predict the future trajectory/velocity of the fly using imaging data
    2. To use more advanced machine learning based methods to do the same


"""

import sklearn.linear_model as lm
from analysis_funs.CX_imaging import CX
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
from scipy import signal as sg
from analysis_funs.utilities import funcs as fn
import pickle
from scipy.optimize import curve_fit,minimize
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
import sklearn.linear_model as lm
from scipy.signal import decimate


class CX_b:
    def __init__(self,datadir,regions):
        self.cxa = CX_a(datadir,regions=regions,denovo=False)
        self.regions = regions
    def prep4RNN(self,twindow,regions,mtype,downsample=False,downfactor=3):
        cxa = self.cxa
        ft2 = cxa.ft2
        pv2 = cxa.pv2
        heading= ft2['ft_heading'].to_numpy()
        tt_o = pv2['relative_time'].to_numpy()
        tt = tt_o
        instrip = ft2['instrip'].to_numpy().astype('float')
        if downsample:
            tt = decimate(tt_o,downfactor)
            heading = np.interp(tt,tt_o,heading)
            
            
            #heading = decimate(heading,downfactor)
            #instrip = decimate(instrip,downfactor)
            instrip = np.interp(tt,tt_o,instrip)
            plt.plot(instrip)
            instrip[instrip<0] = 0
            instrip[instrip>0] = 1
        
        tstep = np.mean(np.diff(tt))
        winlen =np.round( twindow/tstep).astype('int')
        
        #dmod = np.mod(len(heading),winlen)
        
        #heading = heading[:-dmod]
        
        
        
        if mtype=='Phase_amp':
            input_mat = np.zeros((len(heading)-winlen,winlen,3*(len(regions))))
            for ir,r in enumerate(regions):
                phase = cxa.pdat['offset_'+r+'_phase']
                
                twed  = cxa.pdat['wedges_'+r]
                
                ymn_z,pvan_z,pva_z = self.output_amps(twed)
                if downsample:
                    #phase = decimate(phase,downfactor)
                    #ymn_z = decimate(ymn_z,downfactor)
                    #pvan_z = decimate(pvan_z,downfactor)
                    
                    phase = np.interp(tt,tt_o,phase)
                    ymn_z = np.interp(tt,tt_o,ymn_z)
                    pvan_z = np.interp(tt,tt_o,pvan_z)
                    
                for i in range(len(heading)-winlen):
                    tp = phase[i:i+winlen]
                    tmn = ymn_z[i:i+winlen]
                    tc = pvan_z[i:i+winlen]
                    
                    input_mat[i,:,ir*3+0] = tp
                    
                    input_mat[i,:,ir*3+1] = tmn
                    input_mat[i,:,ir*3+2] = tc
        elif mtype =='Phase_amp_plume':
            input_mat = np.zeros((len(heading)-winlen,winlen,3*(len(regions))+2))
            plume = cxa.ft2['instrip'].to_numpy()
            befplume = np.zeros_like(plume)
            pst = np.where(plume>0)[0][0]
            befplume[:pst] =1
            if downsample:
                plume = decimate(plume,downfactor)
                befplume = decimate(befplume,downfactor)
            
            
            for ir,r in enumerate(regions):
                phase = cxa.pdat['offset_'+r+'_phase']
                twed  = cxa.pdat['wedges_'+r]
                ymn_z,pvan_z,pva_z = self.output_amps(twed)
                if downsample:
                    phase = decimate(phase,downfactor)
                    ymn_z = decimate(ymn_z,downfactor)
                    pvan_z = decimate(pvan_z,downfactor)
                    
                for i in range(len(heading)-winlen):
                    tp = phase[i:i+winlen]
                    tmn = ymn_z[i:i+winlen]
                    tc = pvan_z[i:i+winlen]
                    input_mat[i,:,ir*3+0] = tp
                    input_mat[i,:,ir*3+1] = tmn
                    input_mat[i,:,ir*3+2] = tc
                    if ir==0:
                        tpl = plume[i:i+winlen]
                        input_mat[i,:,-1] = tpl
                        tbef = befplume[i:i+winlen]
                        input_mat[i,:,-2] = tbef
            
        self.input_mat =input_mat
        self.y =heading
        self.tt = tt
        self.instrip = instrip
    def reg_traj_model(self,twindow,regions,mtype='Phase_amp',index='all'):
        # Here we are going to make a simple regression based model to predict the 
        # New direction of the fly based upon neural data.
        # Model types:
            # 1. Phase and amplitude models
            # 2. Raw wedges
        cxa = self.cxa
        ft2 = cxa.ft2
        pv2 = cxa.pv2
        heading= ft2['ft_heading'].to_numpy()
        tt = pv2['relative_time'].to_numpy()
        tstep = np.mean(np.diff(tt))
        winlen =np.round( twindow/tstep).astype('int')
        if mtype=='Phase_amp':
            t_phase = np.zeros((len(heading),len(regions)))
            t_amp_mean = np.zeros((len(heading),len(regions)))
            t_amp_pva =np.zeros((len(heading),len(regions)))
            
            for i,r in enumerate(regions):
                tp = cxa.pdat['offset_'+r+'_phase'].to_numpy()
                t_phase[:,i] = tp
                twed  = cxa.pdat['wedges_'+r]
                ymn_z,pvan_z,pva_z = self.output_amps(twed)
                t_amp_mean[:,i] = ymn_z
                t_amp_pva[:,i] = pva_z
            t_phase = (t_phase-np.mean(t_phase,axis=0))/np.std(t_phase,axis=0)
            t_amp_mean = (t_amp_mean-np.mean(t_amp_mean,axis=0))/np.std(t_amp_mean,axis=0)
            t_amp_pva = (t_amp_pva-np.mean(t_amp_pva,axis=0))/np.std(t_amp_pva,axis=0)
            # Make regression matrix
            input_mat = np.ones((len(heading)-winlen,winlen*len(regions)*3+1))
            for i  in range(len(heading)-winlen):
                for ir,r in enumerate(regions):
                    
                    p = t_phase[i:i+winlen,ir]
                   # p = (p-np.mean(p))/np.std(p)
                    pdx = np.arange(ir*winlen,ir*winlen+winlen)
                    input_mat[i,pdx] = p
                    
                    am = t_amp_mean[i:i+winlen,ir]
                    #am = (am-np.mean(am))/np.std(am)
                    adx = np.arange(ir*winlen,ir*winlen+winlen)+len(regions)*winlen
                    input_mat[i,adx] = am
                    
                    av = t_amp_pva[i:i+winlen,ir]
                   # av = (av-np.mean(av)/np.std(av))
                    adx = np.arange(ir*winlen,ir*winlen+winlen)+len(regions)*winlen*2
                    input_mat[i,adx] = av
            
        ymatrix = np.zeros((len(input_mat),2))
        uheading = fn.unwrap(heading)
        uheadings = sg.savgol_filter(uheading,40,3)
        
        dheading = np.diff(uheadings)
        headings = fn.wrap(uheadings)
        headings = headings[winlen:]
        self.input_mat = input_mat
        
        if index =='all':
            dx = np.arange(0,len(headings))
        elif index=='jump_returns':
            jumps = self.cxa.get_jumps(time_threshold=60)
            dx = np.array([],dtype='int')
            for j in jumps:
                d = np.arange(j[1],j[2])
                dx = np.append(dx,d)
        X = input_mat[dx,:]
        y = headings[dx]
        
        self.reg = self.regression_engine(X,y)
        self.y = y
        plt.plot(y)
        yp = np.matmul(X,self.reg.coef_)
        plt.plot(yp)
        r2 = self.reg.score(X,y)
        print(r2)
        #plt.figure()
        #plt.plot(uheading)
        #plt.plot(uheadings)
        #plt.figure()
        #plt.plot(dheading)
                    
        
    ### Try a simple feed forward nn
    
    ### Try a simple RNN
        # Time lagged phase and current amplitude
        # Could build upon asaph's architecture
    
    
    
    def regression_engine(self,x,y):
        reg = lm.LinearRegression(fit_intercept=False)
        reg.fit(x,y)
        return reg
    def output_amps(self,wedges):
        angles = np.linspace(-np.pi,np.pi,16)
        weds = np.sum(wedges*np.sin(angles),axis=1)
        wedc = np.sum(wedges*np.cos(angles),axis=1)
        pva  = np.sqrt(weds**2+wedc**2)
        p0 = np.mean(pva[pva<np.percentile(pva,10)])
        pva = (pva-p0)/p0

        # pva_norm - measure of coherence

        wednorm = wedges
        wednorm = wednorm/np.max(wednorm,axis=1)[:,np.newaxis]

        weds = np.sum(wednorm*np.sin(angles),axis=1)
        wedc = np.sum(wednorm*np.cos(angles),axis=1)
        pva_norm  = np.sqrt(weds**2+wedc**2)
        p0 = np.mean(pva_norm[pva_norm<np.percentile(pva_norm,10)])
        pva_norm = (pva_norm-p0)/p0



        ymn = np.mean(wedges,axis=1)
        y0 = np.mean(ymn[ymn<np.percentile(ymn,10)])
        ymn = (ymn-y0)/y0
        pva_z = pva/np.std(pva)
        pvan_z = pva_norm/np.std(pva_norm)
        ymn_z = ymn/np.std(ymn)
        
        return ymn_z,pvan_z,pva_z
                